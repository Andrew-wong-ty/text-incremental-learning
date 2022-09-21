import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from parsers import args
from typing import List
from utils import load, save
from clean import get_format_train_text

class mydataset(Dataset):
    def __init__(self, dataset:List[dict]):
        self.texts = [item['text'] for item in dataset]
        self.target = np.array([item['target'] for item in dataset])

    def __getitem__(self, index):
        return self.texts[index], self.target[index]
    
    def __len__(self):
        return len(self.texts)

def distribute(oranges, plates):
    base, extra = divmod(oranges, plates)
    arr =  [base + (i < extra) for i in range(plates)]
    arr.sort()
    return arr

def get_IL_dataset(orginal_loader, IL_loader, shuffle):
    # 合并2个data_loader
    orginal_dataset = orginal_loader.dataset
    IL_dataset = IL_loader.dataset
    com_dataset = torch.utils.data.ConcatDataset([orginal_dataset, IL_dataset])
    com_loader = torch.utils.data.DataLoader(dataset=com_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=shuffle)
    return com_loader

def read_wiki80():
    # train = load("/home/tywang/future_study/text-incremental-learning/wiki/train.pkl")
    # dev = load("/home/tywang/future_study/text-incremental-learning/wiki/dev.pkl")
    # test = load("/home/tywang/future_study/text-incremental-learning/wiki/test.pkl")
    # # convert add tags to train, dev and test
    # train = get_format_train_text(train,mode = args.dataset, return_tag=False)
    # #dev = get_format_train_text(dev,mode = args.dataset, return_tag=False)
    # test = get_format_train_text(test,mode = args.dataset, return_tag=False)
    # tags = load("/home/tywang/future_study/text-incremental-learning/wiki/tags.pkl")
    # label2id = load("/home/tywang/future_study/text-incremental-learning/wiki/label2id.pkl")
    # # add keys: label, index
    # train['label'] = [label2id[item] for item in train['rel']]
    # train['index'] = list(range(len(train['text'])))
    # dev['label'] = [label2id[item] for item in dev['rel']]
    # dev['index'] = list(range(len(dev['text'])))
    # test['label'] = [label2id[item] for item in test['rel']]
    # test['index'] = list(range(len(test['text'])))
    # # convert to list
    # keys = train.keys()
    # train_list = []
    # dev_list = []
    # test_list = []
    # for i in range(len(train['text'])):
    #     item = {}
    #     for k in keys:
    #         item[k] = train[k][i]
    #     train_list.append(item)
    # for i in range(len(dev['text'])):
    #     item = {}
    #     for k in keys:
    #         item[k] = dev[k][i]
    #     dev_list.append(item)
    # for i in range(len(test['text'])):
    #     item = {}
    #     for k in keys:
    #         item[k] = test[k][i]
    #     test_list.append(item)
    # train_list.extend(dev_list)
    # save(train_list,"/home/tywang/future_study/text-incremental-learning/wiki/train_list.pkl")
    # # save(dev_list,"/home/tywang/future_study/text-incremental-learning/wiki/dev_list.pkl")
    
    # save(test_list,"/home/tywang/future_study/text-incremental-learning/wiki/test_list.pkl")
    train_list = load("/home/tywang/future_study/text-incremental-learning/wiki/train_list.pkl")
    for idx, item in enumerate(train_list):
        item['index'] = idx
        item['target'] = item['label']
    test_list = load("/home/tywang/future_study/text-incremental-learning/wiki/test_list.pkl")
    for idx, item in enumerate(test_list):
        item['index'] = idx
        item['target'] = item['label']
    tags = load("/home/tywang/future_study/text-incremental-learning/wiki/tags.pkl")


    n_class = len(set([item['target'] for item in train_list]))

    return train_list,test_list,n_class,len(train_list),len(test_list),tags

    debug_stop = 1

def get_Dbpedia():
    data_path = "/data/tywang/dataset/dbpedia_csv"
    train_df = pd.read_csv(os.path.join(data_path,"train.csv"))
    test_df = pd.read_csv(os.path.join(data_path,"test.csv"))
    with open(os.path.join(data_path,"classes.txt"),'r') as file:
        labels = file.readlines()
        labels = [item.replace('\n','') for item in labels]
        N_labels = len(labels)
        id2label = dict(zip([i for i in range(N_labels)], labels))
    # convert csv line to tuple
    train = train_df.apply(lambda x: tuple(x), axis=1).values.tolist() 
    test = test_df.apply(lambda x: tuple(x), axis=1).values.tolist()
    train_dataset = []
    test_dataset = []
    # 需要用到titile吗?
    # for item in train:
    #     train_dataset.append({
    #         'text': item[1]+' [SEP] '+item[2],
    #         'target':item[0]-1
    #     })
    # for item in test:
    #     test_dataset.append({
    #         'text': item[1]+' [SEP] '+item[2],
    #         'target':item[0]-1
    #     })
    for item in train:
        train_dataset.append({
            'text': item[2],
            'target':item[0]-1
        })
    for item in test:
        test_dataset.append({
            'text': item[2],
            'target':item[0]-1
        })
    n_class = len(set([item['target'] for item in train_dataset]))
    print("N_train:{}, N_test:{}, n_class:{}".format(len(train_dataset),len(test_dataset),n_class))
    args.n_classes = n_class
    return train_dataset,test_dataset, n_class, len(train_dataset),len(test_dataset)


def get_IL_loader(train_dataset,test_dataset,n_classes,n_train,n_test):
    train_dataset = mydataset(train_dataset)
    test_dataset = mydataset(test_dataset)
    # check list
    checked_class = []
    checked_train_index = []
    checked_test_index = []
    ## 生成base loader
    # 首先找出是base dataset的index
    
    for idx in range(args.base_class):
        checked_class.append(idx)
        if idx==0:
            target_idx_train = np.where(train_dataset.target==0)[0]
            target_idx_val = np.where(test_dataset.target==0)[0]
        else:
            target_idx_train = np.concatenate([target_idx_train,np.where(train_dataset.target==idx)[0]])
            target_idx_val = np.concatenate([target_idx_val,np.where(test_dataset.target==idx)[0]])
    checked_train_index.extend(target_idx_train.tolist())
    checked_test_index.extend(target_idx_val.tolist())
    dataset_train_base = torch.utils.data.Subset(train_dataset, target_idx_train)
    dataset_val_base = torch.utils.data.Subset(test_dataset, target_idx_val)
    train_loader_base = torch.utils.data.DataLoader(dataset=dataset_train_base,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)
    val_loader_base = torch.utils.data.DataLoader(dataset_val_base,
                                                 batch_size=args.batch_size,
                                                 shuffle=True)
    # incremental learning  的list
    IL_dataset_train = []
    IL_dataset_val = []
    n_classes_for_each_phase = distribute(n_classes - args.base_class, args.phase)
    print("IL class distribution:",n_classes_for_each_phase)
    pointer = args.base_class
    for i, n_each in enumerate(n_classes_for_each_phase):
        # 获取每个phase的需要用到的class 
        for idx in range(pointer, pointer+n_each):
            checked_class.append(idx)
            if idx==pointer:
                target_idx_train = np.where(train_dataset.target==pointer)[0]
                target_idx_val = np.where(test_dataset.target==pointer)[0]
            else:
                target_idx_train = np.concatenate([target_idx_train,np.where(train_dataset.target==idx)[0]])
                target_idx_val = np.concatenate([target_idx_val,np.where(test_dataset.target==idx)[0]])
            
            
        dataset_train_f = torch.utils.data.Subset(train_dataset, target_idx_train)
        dataset_val_f = torch.utils.data.Subset(test_dataset, target_idx_val)
        train_loader_f = torch.utils.data.DataLoader(dataset=dataset_train_f,
                                                            batch_size=args.batch_size,
                                                            shuffle=True)
        val_loader_f = torch.utils.data.DataLoader(dataset_val_f,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)
        IL_dataset_train.append(train_loader_f)
        IL_dataset_val.append(val_loader_f)
        checked_train_index.extend(target_idx_train.tolist())
        checked_test_index.extend(target_idx_val.tolist())
        pointer+=n_each

    # check if all data and classes have been used
    checked_train_index.sort()
    checked_test_index.sort()
    assert checked_class==list(range(n_classes))
    assert checked_train_index==list(range(n_train))
    assert checked_test_index==list(range(n_test))
    return train_loader_base, val_loader_base, IL_dataset_train, IL_dataset_val,n_classes_for_each_phase
if args.dataset=="Dbpedia":
    train_dataset,test_dataset,n_classes,n_train,n_test = get_Dbpedia()
    train_loader_base, val_loader_base, IL_dataset_train, IL_dataset_val,n_classes_for_each_phase = get_IL_loader(
        train_dataset,test_dataset,n_classes,n_train,n_test
    )
    
    debug_stop = 1
if args.dataset=="Wiki80":
    train_dataset,test_dataset,n_classes,n_train,n_test, tags = read_wiki80()
    args.tags = tags
    train_loader_base, val_loader_base, IL_dataset_train, IL_dataset_val,n_classes_for_each_phase = get_IL_loader(
        train_dataset,test_dataset,n_classes,n_train,n_test
    )