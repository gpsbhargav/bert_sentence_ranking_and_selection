import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertConfig, PreTrainedBertModel, BertEmbeddings, BertEncoder, BertPooler

import pdb


class Encoder(PreTrainedBertModel):

    def __init__(self, config):
        super(Encoder, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def construct_binary_mask(self, tensor_in, padding_index = 0):
        '''For bert. 1 denotes actual data and 0 denotes padding'''
        mask = tensor_in != padding_index
        return mask

    def encode(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=False)
        encoded_layers = encoded_layers[-1]

        return encoded_layers

    def forward(self, sequences, segment_id):
        sentence_mask = self.construct_binary_mask(sequences)
        encoding = self.encode(sequences,segment_id, sentence_mask)
        return encoding

class Decoder(PreTrainedBertModel):
        
        def __init__(self, config):
            super(Decoder, self).__init__(config)
            self.encoder = BertEncoder(config)
            # self.pooler = BertPooler(config)
            self.apply(self.init_bert_weights)

        def forward


