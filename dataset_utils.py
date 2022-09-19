import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from parsers import args
from typing import List

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
    checked_idx = []
    checked_train_index = []
    checked_test_index = []
    ## 生成base loader
    # 首先找出是base dataset的index
    
    for idx in range(args.base_class):
        checked_idx.append(idx)
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
            checked_idx.append(idx)
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

    assert checked_idx==list(range(n_classes))
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
    pass