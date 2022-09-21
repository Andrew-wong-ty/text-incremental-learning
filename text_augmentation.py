import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np
import random
eps = 1e-8
import re
import sys

class Augmentation:
    def __init__(self,data):
        self.Dict = self.__get_subj_obj_dict(data)

    def get_augs(self,texts,k):
        """
            拿到[x1,x2,x3..]的k个aug, 返回
            [
                [x1', x2', x3' ...],
                [x1'',x2'',x3''...],
                [...]
            ]
        """
        all_new_texts = [list() for _k in range(k)]
        for text in texts:
            new_texts = self.get_aug(text,k)
            for i in range(k):
                all_new_texts[i].append(new_texts[i])
        return all_new_texts
    def __get_subj_obj_dict(self,data):
        Dict = {'subject':dict(), 'object':dict()}
        all_subj_type = list(set(data['subj_type']))
        all_obj_type = list(set(data['obj_type']))
        self.all_subj_type = all_subj_type
        self.all_obj_type = all_obj_type
        SS = "<S:("+"|".join(all_subj_type)+")>(.*)</S:.*>"
        SO = "<O:("+"|".join(all_obj_type)+")>(.*)</O:.*>"
        CPS = re.compile(SS)
        CPO = re.compile(SO)
        self.CPS = CPS
        self.CPO = CPO
        for subj,subj_type, obj,obj_type in zip(
            data['subj'], data['subj_type'],data['obj'], data['obj_type'] 
        ):
            if subj_type not in Dict['subject']:
                Dict['subject'][subj_type] = [subj]
            else:
                Dict['subject'][subj_type].append(subj)
            
            if obj_type not in Dict['object']:
                Dict['object'][obj_type] = [obj]
            else:
                Dict['object'][obj_type].append(obj)
        return Dict
    def get_aug(self,text,k = 1):
        """
            针对这一个text, 拿到k个aug, 以列表返回
        """
        try:
            subj_search_res = self.CPS.search(text)
            obj_search_res = self.CPO.search(text)
            try:
                subj_start, subj_end = subj_search_res.span(2) 
                parse_subjtype, subj = subj_search_res.groups()
                assert parse_subjtype.strip() in self.all_subj_type
            except:
               subj_start, subj_end =  (None,None)
               parse_subjtype, subj = (None, None)
            try:
                obj_start, obj_end = obj_search_res.span(2)
                parse_objtype, obj = obj_search_res.groups()
                assert parse_objtype.strip() in self.all_obj_type
            except:
                obj_start, obj_end = (None, None)
                parse_objtype, obj =  (None, None)
            
            assert subj_start is not None or obj_start is not None
            
        except:
            print(text)
            sys.exit()

        if subj_start is None or obj_start is None:
            if subj_start is None:
                try:
                    random_entity = random.choices(self.Dict['object'][parse_objtype],k=k)
                except:
                    print(text)
                    sys.exit()
                start,end = obj_start, obj_end
            else:
                try:
                    random_entity = random.choices(self.Dict['subject'][parse_subjtype],k=k)
                except:
                    print(text)
                    sys.exit()
                start,end = subj_start, subj_end
        else:
            rn = random.random()
            if rn>0.5:
                # 换subj的
                try:
                    random_entity = random.choices(self.Dict['subject'][parse_subjtype],k=k)
                except:
                    print(text)
                    sys.exit()
                start,end = subj_start, subj_end
            else:
                # 换obj的
                try:
                    random_entity = random.choices(self.Dict['object'][parse_objtype],k=k)
                except:
                    print(text)
                    sys.exit()
                start,end = obj_start, obj_end
        texts = []
        for entity in random_entity:
            aug_text = text[:start]+" "+entity+" "+text[end:]
            texts.append(aug_text)
        return texts
