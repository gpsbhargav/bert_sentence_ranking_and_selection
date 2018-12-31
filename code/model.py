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
        self.query_for_sentence_ranking = nn.Parameter(torch.randn(1, 1, options.bert_hidden_size, requires_grad=True))
        
    
    def construct_binary_mask(self, tensor_in, padding_index = 0):
        '''For bert. 1 denotes actual data and 0 denotes padding'''
        mask = tensor_in != padding_index
        return mask

    def construct_additive_mask(self, tensor_in, padding_index = 0):
        mask = tensor_in == padding_index
        float_mask = mask.type(dtype=torch.float32)
        float_mask =  float_mask.masked_fill(mask=mask, value=-1e-20)
        return float_mask
    
    def forward(self, data):
        sentence_reps = []
        for i in range(self.options.max_para_len):
            sentence_mask = self.construct_binary_mask(data["sequence_{}".format(i)])
            _, pooled_representation = self.bert(data["sequence_{}".format(i)],
                                                                 token_type_ids=data["sequence_segment_id_{}".format(i)],
                                                                 attention_mask=sentence_mask, 
                                                                 output_all_encoded_layers=False)
            sentence_reps.append(pooled_representation)
        
        
        paragraph_rep = torch.stack(sentence_reps,dim=1)
        
        # paragraph_rep = self.options.dropout(paragraph_rep)
        
        raw_scores = torch.sum(self.query_for_sentence_ranking * paragraph_rep, dim=-1)
        
        passage_mask = self.construct_additive_mask(data["passage_mask"])
        
        raw_scores = raw_scores + passage_mask
        
        return raw_scores
