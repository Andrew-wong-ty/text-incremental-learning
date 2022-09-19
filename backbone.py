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


class BertClassifier(BertPreTrainedModel):
    def __init__(self, model, num_labels:int, args):
        config = AutoConfig.from_pretrained(model)
        super().__init__(config)
        self.args = args
        self.num_labels = num_labels
        self.config = config

        self.bert = AutoModel.from_config(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model) 
        try:
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
        except:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.fc = nn.Linear(config.hidden_size, self.num_labels, bias = False)

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

    def forward(self,text_arr):
        text_feat = self.tokenize(text_arr)

        # freeze the parameters of bert backbone
        outputs = self.bert(**text_feat)
        # outputs = self.bert(
        #     text_feat['input_ids'],
        #     attention_mask=text_feat['attention_mask'],
        #     token_type_ids=text_feat['token_type_ids'] if hasattr(text_feat,'token_type_ids') else None,
        #     output_attentions=True,
        #     output_hidden_states=True,
        # )
        # hidden_states=outputs.hidden_states,
        # attentions=outputs.attentions,
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)

        return logits



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