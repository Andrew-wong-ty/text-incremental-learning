import random
import torch
from torch import nn
import numpy as np
import transformers
import  pickle
from enum import Enum
import time
import torch.nn.functional as F
from parsers import args
from text_augmentation import Augmentation

def augment_text(text_arr,Aug:Augmentation):
    return [Aug.get_aug(text)[0] for text in text_arr]

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,args, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.device = torch.device("cuda:{}".format(args.cuda_index))
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, Xi, Xt, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = torch.cat([Xi.unsqueeze(1), Xt.unsqueeze(1)], dim=1)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
            features = F.normalize(features,dim = 2)  # 加多一个normalize的操作
        else:
            features = F.normalize(features,dim = -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # 将 Xi和Xt 2 个矩阵上下拼接
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature # 将 Xi和Xt 2 个矩阵上下拼接
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits = logits.to(self.device)
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # 去掉logits上的对角线自相乘
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveLoss(nn.Module):
    """
        implementation of ContrastiveLoss 
    """
    def __init__(self,temp:float = 0.07):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
    def forward(self, Xi, Xt):
        """calculate the contrastive loss of X and Y, which are 3D tensors with shape (bs, max_len, d_model)

            Args:
                Xi: the feature of image
                Xt: the feature of text
            return:
                Contrastive loss of Xi and Xt
        """
        # Xi dot (Xt^T)
        similarity_i2t = Xi @ Xt.transpose(1,2) / self.temp
        # create mask
        mask = torch.zeros_like(similarity_i2t)
        for i in range(len(mask)):mask[i] = mask[i].fill_diagonal_(1)
        # calculate loss
        # 这里应该是搞错了, 这里的CL loss是Xi和Xt的某个样本的channel之间做CL, 但其实, 我们需要的是不同的样本之间做CL
        loss_pre_sample = -torch.sum(F.log_softmax(similarity_i2t,dim=-1)*mask, dim=-1).mean(-1)
        loss = loss_pre_sample.mean()
        return  loss





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



if __name__=="__main__":
    # batch_size = 32
    # d_model = 128
    # x = torch.rand(batch_size, d_model).float()
    # x_p = torch.rand(batch_size, d_model).float()
    CLoss = SupConLoss(args)
    # loss = CLoss(x, x_p)

    a = torch.tensor([
        [1.,2.],
        [3.,4.],
        [5.,6.]
    ])
    b = torch.tensor([
        [1.,2.],
        [3.,4.],
        [5.,6.]
    ])
    loss = CLoss(a, b)
    c = torch.cat([a.unsqueeze(1), b.unsqueeze(1)], dim=1)
    c = F.normalize(c,dim=-1)
    debug_stop = 1