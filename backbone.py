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

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoConfig, BertForSequenceClassification,BertPreTrainedModel\
    ,AutoModel,AutoTokenizer 
from utils import get_subj_obj_start

class BertClassifier(BertPreTrainedModel):
    def __init__(self, model, num_labels:int, args):
        config = AutoConfig.from_pretrained(model)
        super().__init__(config)
        self.args = args
        self.num_labels = num_labels
        self.config = config

        self.bert = AutoModel.from_config(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model) 
        
        if args.dataset=="Wiki80":
            self.additional_index = len(self.tokenizer) 
            if len(args.tags)!=0:
                print("Add {num} special tokens".format(num=len(args.tags)))
                special_tokens_dict = {'additional_special_tokens': args.tags}
                self.tokenizer.add_special_tokens(special_tokens_dict)
                self.bert.resize_token_embeddings(len(self.tokenizer))  
        try:
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
        except:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        if args.dataset=="Wiki80":
            self.fc = nn.Linear(config.hidden_size*2, self.num_labels, bias = False)
            #self.CL_project_head = nn.Linear(config.hidden_size*2,config.hidden_size*2)
        else:
            self.fc = nn.Linear(config.hidden_size, self.num_labels, bias = False)
            #self.CL_project_head = nn.Linear(config.hidden_size,config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()
    
    def tokenize(self, text_arr):
        if isinstance(text_arr,tuple):
            text_arr = list(text_arr)
        text_feat= self.tokenizer.batch_encode_plus(text_arr, 
                                                    max_length=self.args.max_length, 
                                                    return_tensors='pt', 
                                                    padding='longest',
                                                    truncation=True)
        for k,_ in text_feat.items():
            text_feat[k] = text_feat[k].to(self.args.device)
        return text_feat


    def get_embedding_PURE(self, text_feat):
        ent1_spos,ent2_spos = get_subj_obj_start(text_feat['input_ids'],self.tokenizer,self.additional_index)
        outputs = self.bert(**text_feat)
        last_hidden_state = outputs[0]
        bs = last_hidden_state.shape[0]
        assert len(ent1_spos)==len(ent2_spos)
        ent1_spos = torch.tensor(ent1_spos)
        ent2_spos = torch.tensor(ent2_spos)
        embedding1 = last_hidden_state[[i for i in range(bs)],ent1_spos,:] 
        embedding2 = last_hidden_state[[i for i in range(bs)],ent2_spos,:] 
        embeddings = torch.cat([embedding1,embedding2],dim = 1)
        # if self.args.use_closs:
        #     feat1 = self.CL_project_head(embedding1)
        #     feat2 = self.CL_project_head(embedding2)
        return embeddings  

    def get_embedding(self, text_feat):
        outputs = self.bert(**text_feat)
        pooled_output = outputs[1]
        return pooled_output
        
    def forward(self,text_arr):
        text_feat = self.tokenize(text_arr)

        # outputs = self.bert(**text_feat)
        # pooled_output = outputs[1]
        pooled_output = self.get_embedding_PURE(text_feat) if self.args.dataset=="Wiki80" else self.get_embedding(text_feat)

        pooled_output_dropout = self.dropout(pooled_output)
        logits = self.fc(pooled_output_dropout)

        return logits,pooled_output



if __name__ == '__main__':
    class Argument:
        def __init__(self):
            self.device = torch.device("cuda:0")
            self.max_length = 128

    args = Argument()
    bert = BertClassifier("/data/transformers/bert-base-uncased",10,args).to(args.device)
    logits = bert.forward(['Hi, good morning.','hello, my friend.'])
    print(bert)
    a = torch.rand(4,3)
    index = torch.argmax(a,-1)
    debug_stop = 1