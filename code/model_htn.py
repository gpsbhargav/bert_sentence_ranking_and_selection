import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertConfig, PreTrainedBertModel, BertEmbeddings, BertEncoder, BertPooler, BertModel

import pdb


class Encoder(PreTrainedBertModel):

    def __init__(self, config, options):
        super(Encoder, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)
        self.options = options
        self.dropout = nn.Dropout(p=options.dropout, inplace=False)
        self.fc_1 = nn.Linear(options.bert_hidden_size*2, options.bert_hidden_size, bias=True)

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

    def forward(self, sequences, segment_id, start_index, end_index):
        sentence_mask = self.construct_binary_mask(sequences)
        encoding = self.encode(sequences,segment_id, sentence_mask)
        
        paragraph_encoding = encoding.permute(0,2,1)
        start_vectors = torch.bmm(paragraph_encoding, start_index.permute(0,2,1)).permute(0,2,1)
        end_vectors = torch.bmm(paragraph_encoding, end_index.permute(0,2,1)).permute(0,2,1)

        start_end_concatenated = torch.cat([start_vectors, end_vectors], dim=-1)
        start_end_concatenated = self.dropout(start_end_concatenated)
        
        sentence_reps = self.fc_1(start_end_concatenated)

        return sentence_reps


class BERT(nn.Module):
    
    def __init__(self, options):
        super(BERT, self).__init__()
        self.options = options
        self.bert = BertModel.from_pretrained(options.bert_type, cache_dir=options.bert_archive)
        self.dropout = nn.Dropout(p=options.dropout, inplace=False)
        self.fc_1 = nn.Linear(options.bert_hidden_size*2, options.bert_hidden_size, bias=True)
        
    
    def construct_binary_mask(self, tensor_in, padding_index = 0):
        '''For bert. 1 denotes actual data and 0 denotes padding'''
        mask = tensor_in != padding_index
        return mask

    def construct_additive_mask(self, tensor_in, padding_index = 0):
        mask = tensor_in == padding_index
        float_mask = mask.type(dtype=torch.float32)
        float_mask =  float_mask.masked_fill(mask=mask, value=-1e-20)
        return float_mask
    
    def forward(self, sequences, segment_id, start_index, end_index):

        sentence_mask = self.construct_binary_mask(sequences)
        
        paragraph_encoding, _ = self.bert(sequences,token_type_ids=segment_id,attention_mask=sentence_mask, output_all_encoded_layers=False)
        
        paragraph_encoding = paragraph_encoding.permute(0,2,1)
        start_vectors = torch.bmm(paragraph_encoding, start_index.permute(0,2,1)).permute(0,2,1)
        end_vectors = torch.bmm(paragraph_encoding, end_index.permute(0,2,1)).permute(0,2,1)

        start_end_concatenated = torch.cat([start_vectors, end_vectors], dim=-1)
                
        start_end_concatenated = self.dropout(start_end_concatenated)

        return start_end_concatenated



class Decoder(PreTrainedBertModel):
        
        def __init__(self, config, options):
            super(Decoder, self).__init__(config)
            self.encoder = BertEncoder(config)
            self.apply(self.init_bert_weights)
            self.options = options
            self.fc_1 = nn.Linear(options.bert_hidden_size*2, options.bert_hidden_size, bias=True)
            self.fc_2 = nn.Linear(options.bert_hidden_size, 1, bias=True)

        def forward(self, vectors_in, attention_mask=None):
            if attention_mask is None:
                attention_mask = torch.ones(vectors_in.shape[0],vectors_in.shape[1], device=vectors_in.device)
            
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            
            vectors_in = self.fc_1(vectors_in)
            encoded_layers = self.encoder(vectors_in,
                                      extended_attention_mask,
                                      output_all_encoded_layers=False)
            encoded_layers = encoded_layers[-1]
            
            raw_scores = self.fc_2(encoded_layers).squeeze(dim=-1)

            return raw_scores






