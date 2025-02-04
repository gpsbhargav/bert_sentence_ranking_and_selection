import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertModel

import pdb



class SentenceSelector(nn.Module):
    
    def __init__(self, options, device):
        super(SentenceSelector, self).__init__()
        self.options = options
        self.device = device
        self.bert = BertModel.from_pretrained(options.bert_type, cache_dir=options.bert_archive)
        self.dropout = nn.Dropout(p=options.dropout, inplace=False)
        self.fc_1 = nn.Linear(options.bert_hidden_size, 1, bias=True)
        
    
    def construct_binary_mask(self, tensor_in, padding_index = 0):
        '''For bert. 1 denotes actual data and 0 denotes padding'''
        mask = tensor_in != padding_index
        return mask

    def construct_additive_mask(self, tensor_in, padding_index = 0):
        mask = tensor_in == padding_index
        float_mask = mask.type(dtype=torch.float32)
        float_mask =  float_mask.masked_fill(mask=mask, value=-1e-20)
        return float_mask
    
    def forward(self, sequences, segment_id):

        sentence_mask = self.construct_binary_mask(sequences)
        
        _, pooled_representation = self.bert(sequences,token_type_ids=segment_id,attention_mask=sentence_mask, output_all_encoded_layers=False)
        
        paragraph_rep = pooled_representation
        
        paragraph_rep = self.dropout(paragraph_rep)
        
        raw_scores = self.fc_1(paragraph_rep).squeeze(dim=-1)

        return raw_scores
