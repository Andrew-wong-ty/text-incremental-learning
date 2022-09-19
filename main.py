import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(CURR_DIR)
P = PATH.parent
for i in range(3):
    P = P.parent
    sys.path.append(str(P.absolute()))
import time
TIME=time.strftime("%m-%d-%H-%M-%S", time.localtime())
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from backbone import BertClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

from parsers import args
from utils import set_global_random_seed, AverageMeter, Summary, ProgressMeter, get_time, save, load
import numpy as np
import tqdm
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

def adjust_learning_rate():
    pass

def cal_acc(true, predict):
    N_right = 0
    assert len(true)==len(predict)
    for t, p in zip(true, predict):
        if t==p:
            N_right+=1
    return N_right/len(true)

def precision_recall_f1_acc(target, output, average = 'macro'):
    output = output.detach().cpu()
    predicted_label = torch.argmax(output,-1).numpy()
    gt = target.numpy()
    precision,recall,f1,_ = precision_recall_fscore_support(gt, predicted_label,average=average)
    acc = accuracy_score(gt, predicted_label)
    return precision,recall,f1,acc


def train(train_loader, model:BertClassifier, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    accs = AverageMeter('acc', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(train_loader),
        [losses, accs],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    for i_batch, (text, target) in enumerate(train_loader):
        adjust_learning_rate() 
        output = model(text)
        loss = criterion(output, target.to(args.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # calculate metric
        _,_,_,acc = precision_recall_f1_acc(target, output)
        losses.update(loss.item(), len(text))
        accs.update(acc, len(text))

        if i_batch % args.print_freq == 0:
            progress.display(i_batch)

        # if i_batch==400:
        #     # for debug
        #     break

@torch.no_grad()
def validate(val_loader, model:BertClassifier, criterion, args):
    model.eval()
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    accs = AverageMeter('acc', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [losses, accs],
        prefix='Test: ')
    gts = []
    pre = []
    model.eval()
    for i_batch, (text, target) in enumerate(val_loader):
        output = model(text)
        loss = criterion(output, target.to(args.device))
        # calculate metric
        predicted_label = torch.argmax(output.detach().cpu(),-1).tolist()
        gt = target.tolist()
        gts.extend(gt)
        pre.extend(predicted_label)
        _,_,_,acc = precision_recall_f1_acc(target, output)
        losses.update(loss.item(), len(text))
        accs.update(acc, len(text))
        if i_batch % args.print_freq == 0:
            progress.display(i_batch)
    precision,recall,f1,_ = precision_recall_fscore_support(gts, pre,average='macro')
    acc = accuracy_score(gts, pre)
    current_time = get_time()
    save({"gt":gts,"pre":pre},"/home/tywang/future_study/text-incremental-learning/test_output/eval_nclasses{}_{}.pkl".format(len(set(gts)),current_time))
    print("Eval acc:{} f1:{}".format(acc,f1))
    print("eval distribution: gts:",Counter(gts),"  pres:",Counter(pre))
    return acc,f1

def cls_align(train_loader, wrapped_model, args):
    model = wrapped_model
    # 最后的2层 fc, 映射到W_fe
    new_model_fc = torch.nn.Sequential(
        model.fc[:2]
    )
    model.eval()

    auto_cor = torch.zeros(model.fc[-1].weight.size(1), model.fc[-1].weight.size(1)).to(args.device, non_blocking=True)
    crs_cor = torch.zeros(model.fc[-1].weight.size(1), args.num_classes).to(args.device, non_blocking=True)
    with torch.no_grad():
        for epoch in range(1):
            pbar = tqdm.tqdm(enumerate(train_loader), desc='Re-Alignment Base', total=len(train_loader), unit='batch')
            for i, (text, target) in pbar:
                
                # get new activation
                text_feat = model.tokenize(text)
                outputs = model.bert(**text_feat)
                pooled_output = outputs[1]
                pooled_output = model.dropout(pooled_output)
                new_activation = new_model_fc(pooled_output)
                #

                # new_activation = new_model((text_feat['input_ids'],text_feat['attention_mask']))
                label_onehot = F.one_hot(target, args.num_classes).float().to(args.device, non_blocking=True)
                auto_cor += torch.t(new_activation) @ new_activation
                crs_cor += torch.t(new_activation) @ (label_onehot)
    print('numpy inverse')
    R = np.mat(auto_cor.cpu().numpy() + args.rg * np.eye(model.fc[-1].weight.size(1))).I
    R = torch.tensor(R).float().to(args.device, non_blocking=True)
    Delta = R @ crs_cor
    model.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(0.9*Delta.float()))
    return R


def IL_align(train_loader, model, args, R, repeat = 1):
    new_model_fc = torch.nn.Sequential(
        model.fc[:2]
    )
    model.eval()
    W = (model.fc[-1].weight.t()).double()
    R = R.double()
    with torch.no_grad():
        for epoch in range(repeat):
            pbar = tqdm.tqdm(enumerate(train_loader), desc='Re-Alignment', total=len(train_loader), unit='batch')
            for i, (text, target) in pbar:
                
                # get new activation
                text_feat = model.tokenize(text)
                outputs = model.bert(**text_feat)
                pooled_output = outputs[1]
                pooled_output = model.dropout(pooled_output)
                new_activation = new_model_fc(pooled_output)
                new_activation = new_activation.double()
                #

                label_onehot = F.one_hot(target, args.num_classes).double().to(args.device, non_blocking=True)
                R = R - R@new_activation.t()@torch.pinverse(torch.eye(len(text)).to(args.device, non_blocking=True) +
                                                                    new_activation@R@new_activation.t())@new_activation@R
                W = W + R @ new_activation.t() @ (label_onehot - new_activation @ W)
    model.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(W.float()))
    return R


def main():
    print(args)
    set_global_random_seed(args.seed)
    args.device = torch.device("cuda:{}".format(args.cuda_index))
    from dataset_utils import train_loader_base, val_loader_base, IL_dataset_train, IL_dataset_val ,n_classes_for_each_phase
    train_loader, val_loader = train_loader_base, val_loader_base
    model = BertClassifier(args.model,args.base_class,args).to(args.device)
    print(model)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # base training 
    best_f1 = -1
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)
        acc,f1 = validate(val_loader, model, criterion, args)
        if f1>best_f1:
            best_f1 = f1
            torch.save(
                {
                'epoch': epoch + 1,
                'args': str(args),
                'state_dict': model.state_dict(),
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict(),
                }
                , os.path.join(args.model_save_path, "checkpoint_{}.pth".format(args.dataset)))

    # load model for debug
    # model.load_state_dict(torch.load(os.path.join(args.model_save_path, "checkpoint_{}.pth".format(args.dataset)))['state_dict'])
    # acc,f1 = validate(val_loader, model, criterion, args)
    # print("acc and f1 after loading trained model, acc:{:.4f}, f1:{:.4f}".format(acc,f1))
    #
    # increment learning 
    bias_fe = False
    model.fc = nn.Sequential(nn.Linear(model.fc.weight.size(1), args.Hidden, bias=bias_fe),
                                    nn.ReLU(),
                                    nn.Linear(args.Hidden, args.base_class, bias=False)).to(args.device)

    f1_cil = []
    print('Training for base classes...')
    # cls re-alignment for base
    args.num_classes = args.base_class
    R = cls_align(train_loader, model, args)
    acc,f1 = validate(val_loader, model, criterion, args)
    print("acc and f1 after alignment, acc:{:.4f}, f1:{:.4f}".format(acc,f1))

    #
    # f1_cil.append(f1)
    # print('Base phase {}/{}: acc:{:.4f}%, f1:{:.4f}'.format(0, args.phase, acc,f1))

    # CIL for phases
    print('Training for incremental classes with {} phase(s) in total...'.format(args.phase))
    from dataset_utils import get_IL_dataset
    for phase in range(args.phase):
        args.num_classes = args.base_class + sum(n_classes_for_each_phase[:phase+1])
        W = model.fc[-1].weight
        W = torch.cat([W, torch.zeros(n_classes_for_each_phase[phase], args.Hidden).to(args.device)], dim=0)
        model.fc[-1] = nn.Linear(args.Hidden, args.num_classes, bias=False)
        model.fc[-1].weight = torch.nn.parameter.Parameter(W.float())
        R = IL_align(IL_dataset_train[phase], model, args, R, args.repeat)
        print('Incremental Learning for Phase {}/{}'.format(phase + 1, args.phase))
        # evaluate new model
        val_loader = get_IL_dataset(val_loader, IL_dataset_val[phase], False)
        acc,f1 = validate(val_loader, model, criterion, args)
        f1_cil.append(f1)
        print('Phase {}/{}: acc:{:.4f}%, f1:{:.4f}'.format(phase+1,args.phase,acc,f1))
    avg = sum(f1_cil)/len(f1_cil)
    print('The average accuracy: {}'.format(avg))


if __name__ == '__main__':
    main()