import random
import torch
import numpy as np
import transformers
import  pickle
from enum import Enum
import time




def get_subj_obj_start(input_ids_arr,tokenizer,additional_index):
    """
    Function:
        to find the positon of the additional token, for example, 
        suppose we have a text: '<S:PERSON> Jobs </S:PERSON> is the founder of <O:ORGANIZATION> Apple </O:ORGANIZATION>'
        we gonna find the index of '<S:PERSON>' and '<O:ORGANIZATION>'
    Args:
    input_ids_arr like:
        tensor([[  101,  9499,  1071,  2149, 30522,  8696, 30522, 30534,  6874,  9033,
            4877,  3762, 30534, 10650,  1999, 12867,  1024,  5160,   102,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0],
            [  101,  2019, 21931, 17680,  2013, 11587, 30532,  2149, 30532, 14344,
            5016, 30537,  2406, 22517,  3361, 30537,  2006,  5958,  1010, 11211,
            2007, 10908,  2005,  1037,  2149,  3446,  3013,  2006,  9317,  1010,
            2992,  8069,  2008,  1996, 23902,  2013,  1996,  2149,  3847, 24185,
            2229,  2003, 24070,  1010, 16743,  2056,  1012,   102]])
    tokenizer:
        as named
    additional_index:
        the first index of the additional_special_tokens
    return:
         subj and obj start position
    """
    subj_starts = []
    obj_starts = []
    for input_ids in input_ids_arr:
        
        subj_start = -1
        obj_start = -1
        checked_id = []
        for idx,word_id in enumerate(input_ids):
            if subj_start!=-1 and obj_start!=-1:
                break
            if word_id>=additional_index:
                if word_id not in checked_id:
                    checked_id.append(word_id)
                    decoded_word = tokenizer.decode(word_id)
                    if decoded_word.startswith("<S:"):
                        subj_start = idx
                    elif decoded_word.startswith("<O:"):
                        obj_start = idx
        if subj_start==-1 or obj_start==-1:
            
            
            if subj_start==-1:
                subj_start=0
            if obj_start==-1:
                obj_start=0
        subj_starts.append(subj_start)
        obj_starts.append(obj_start)
    return subj_starts,obj_starts

def get_time():
    curr_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    return curr_time

def save(obj,path_name):
    print("save to:",path_name)
    with open(path_name,'wb') as file:
        pickle.dump(obj,file)

def load(path_name: object) -> object:
    
    with open(path_name,'rb') as file:
        return pickle.load(file)

def set_global_random_seed(seed):
    print("set seed:",seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)