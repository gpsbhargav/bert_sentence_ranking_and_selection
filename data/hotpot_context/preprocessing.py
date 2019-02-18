#!/usr/bin/env python
# coding: utf-8

# # Preprocess hotpotqa for BERT
# - Input to BERT will be question and a paragraph
# - All sentences of the paragraph will be concatenated (possibly separated by [SEP])
# - Labels will be binary vector
# - The indices of the supporting fact vectors to extract will be supplied as a binary matrix. Multiplying by this matrix will extract the required vectors.

# In[ ]:


from collections import Counter
import string
import re
import argparse
import json
import sys
import numpy as np
import nltk
import random
import math
import os
import pickle
from tqdm import tqdm, trange

import pdb


# In[ ]:


from pytorch_pretrained_bert import BertTokenizer


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[ ]:


def pickler(path,pkl_name,obj):
    with open(os.path.join(path, pkl_name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def unpickler(path,pkl_name):
    with open(os.path.join(path, pkl_name) ,'rb') as f:
        obj = pickle.load(f)
    return obj


# In[ ]:


TRAINING = True

out_pkl_path = "./"

if(TRAINING):
    file_path = "/home/bhargav/data/hotpotqa/hotpot_train_v1.json"
    out_pkl_name = "preprocessed_train.pkl"
    small_out_pkl_name = "preprocessed_train_small.pkl"
    small_dataset_size = 5000
    problem_indices = [8437, 25197, 34122, 46031, 52955, 63867, 82250]
else:
    file_path = "/home/bhargav/data/hotpotqa/hotpot_dev_distractor_v1.json"
    out_pkl_name = "preprocessed_dev.pkl"
    small_out_pkl_name = "preprocessed_dev_small.pkl"
    small_dataset_size = 500
    problem_indices = [5059]
    
    

# max_seq_len = 501  # Final sequence length will be max_seq_len + (max_num_paragraphs - 1) = 510
max_seq_len = 500
max_sentences = 5 
max_num_paragraphs = 10


# In[ ]:


with open(file_path, encoding='utf8') as file:
    dataset = json.load(file)


# In[ ]:


def tokenize(text):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


# In[ ]:


question_ids = []
questions = []
paragraphs = [] 
supporting_facts = []

skipped = []

for item_index, item in enumerate(tqdm(dataset)):
    if(item_index in problem_indices):
        skipped.append(item_index)
        continue
    if(len(item["context"]) != 10):
        skipped.append(item_index)
        continue
    question_ids.append(item["_id"])
    question = tokenize(item["question"])
    questions.append(question)
    paragraph_names = []
    paragraph_text = []
    for i,para in enumerate(item["context"]):
        para_name = para[0]
        para_sents = para[1]
        paragraph_names.append(para_name)
        paragraph_text.append([tokenize(s) for s in para_sents])
    paragraphs.append(paragraph_text)
    supp_fact_list = []
    for sup_fact in item["supporting_facts"]:
        para_name = sup_fact[0]
        supporting_fact_index = sup_fact[1] 
        para_index = paragraph_names.index(para_name)
        supp_fact_list.append([para_index, supporting_fact_index])
    supporting_facts.append(supp_fact_list)


# In[ ]:


print("Skipped {} records".format(len(skipped)))


# # TODO
# - Merge all sentences in a paragraph
# - Merge question with the above and add [CLS]
# - Pad all sequences to fixed length (512 is BERT's limit)
# - Fix the number of sentences in each para
# - form the supporting fact indicator vector
# - form the matrix using which the start and end vector of each sentence can be extracted. Use the index of [PAD] for making sure each para has the same amount of sentences
# - write pkl

# In[ ]:


cls_index = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
sep_index = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
pad_index = tokenizer.convert_tokens_to_ids(["[PAD]"])[0]


# In[ ]:


def trim_paragraph(paragraph, max_seq_len):
    assert(max_seq_len >= 0)
    sent_lengths = [len(s) for s in paragraph]
    out_paragraph = []
    length_so_far = 0
    for sent in paragraph:
        if(len(sent) == 0):
            continue
        if(length_so_far + len(sent) <= max_seq_len):
            out_paragraph.append(sent)
            if(length_so_far + len(sent) == max_seq_len):
                break
            length_so_far += len(sent)      
        else:
            sent = sent[:max_seq_len-length_so_far]
            out_paragraph.append(sent)
            break
    return out_paragraph
    
def pad_paragraph(paragraph, max_sequence_len, pad_index):
    assert(max_sequence_len >= 0)
    sent_lengths = [len(s) for s in paragraph]
    assert(sum(sent_lengths) <= max_sequence_len)
    paragraph[-1] += [pad_index] * (max_sequence_len - sum(sent_lengths))
    return paragraph

def merge_trim_pad_paragraphs(question, paragraph, paragraph_index, supporting_facts_in, max_seq_len, max_sentences, 
                   cls_index, sep_index, pad_index):
    sentence_start_indices = []
    sentence_end_indices = []
    
    paragraph = paragraph[:max_sentences]
    
    total_para_len_words = sum([len(s) for s in paragraph])
    
    available_length_for_paragraph = max_seq_len - (len(question) + 2) # question + CLS + SEP
    
    if(total_para_len_words >= available_length_for_paragraph):
        paragraph = trim_paragraph(paragraph, available_length_for_paragraph-1) #-1 to make room for the next empty sentence
        paragraph[-1].append(pad_index)
    elif(total_para_len_words < available_length_for_paragraph):
        paragraph = pad_paragraph(paragraph, available_length_for_paragraph, pad_index)
        
        
    #concatenate sentences, note starting and ending indices of sentences
    sentence_start_indices = []
    sentence_end_indices = []
    out_sequence = [cls_index] + question + [sep_index]
    for sent in paragraph:
        sentence_start_indices.append(len(out_sequence))
        out_sequence += sent
        sentence_end_indices.append(len(out_sequence)-1)
    
    assert(len(sentence_start_indices) == len(sentence_end_indices))
    
    #create the matrix used to extract first and last vectors of sentences
#     start_index_matrix = []
#     end_index_matrix = []
#     for i in range(len(sentence_start_indices)):
#         start_indicator_vector = [0] * max_seq_len
#         end_indicator_vector = [0] * max_seq_len
#         start_indicator_vector[sentence_start_indices[i]] = 1
#         end_indicator_vector[sentence_end_indices[i]] = 1
#         start_index_matrix.append(start_indicator_vector)
#         end_index_matrix.append(end_indicator_vector)
        
    #create supporting_facts vector
    supporting_facts = [0] * max_sentences
    for s_f in supporting_facts_in:
        if(s_f[0] == paragraph_index and s_f[1]<max_sentences):
            supporting_facts[s_f[1]] = 1
            
    #expand start and end index matrices to make sure all extract equal number of vectors (=max_sentences)
#     if(max_sentences - len(paragraph) > 0):
#         fake_sentence_start_and_end = [0] * max_seq_len
#         index_of_pad = out_sequence.index(pad_index)
#         fake_sentence_start_and_end[index_of_pad] = 1
#         for i in range(max_sentences - len(paragraph)):
#             start_index_matrix.append(fake_sentence_start_and_end)
#             end_index_matrix.append(fake_sentence_start_and_end)
            
    segment_id = [0]*(len(question) + 2)
    segment_id += [1]*(max_seq_len - (len(question) + 2))
    
    # sanity check
    assert(len(out_sequence) == max_seq_len)
    assert(len(segment_id) == max_seq_len)
#     assert(len(start_index_matrix) == max_sentences)
#     assert(len(end_index_matrix) == max_sentences)
    assert(len(supporting_facts) == max_sentences)
    
    return {'sequence': out_sequence, #'start_index':start_index_matrix, 'end_index':end_index_matrix, 
            'sentence_start_index': sentence_start_indices, 'sentence_end_index': sentence_end_indices,
            'supporting_fact': supporting_facts, 'segment_id':segment_id}
    


# In[ ]:


out_dict = {'sequence': [], 'sentence_start_index':[], 'sentence_end_index':[], 'supporting_fact': [], 'segment_id': [],
           'max_seq_len':max_seq_len, 'max_sentences':max_sentences, "question_id":[]}
q_count = 0
p_count = 0
for i,q in enumerate(tqdm(questions)):
    q_count+=1
    for j,para in enumerate(paragraphs[i]):
        p_count += 1
        processed_example = merge_trim_pad_paragraphs(question=q, paragraph=para, paragraph_index=j, 
                                                      supporting_facts_in=supporting_facts[i], 
                                                      max_seq_len=max_seq_len, max_sentences=max_sentences, 
                                                      cls_index=cls_index, sep_index=sep_index, pad_index=pad_index)
        out_dict["question_id"].append(question_ids[i])
        for key,value in processed_example.items():
            out_dict[key].append(value)


# In[ ]:


Counter([len(p) for p in paragraphs])


# In[ ]:


print(q_count)
print(p_count)


# In[ ]:


# save number of sentences per document and number of paragraphs per document so that the predictions can be 
# reformatted to the standard format
document_lengths = []

for doc in paragraphs:
    num_sentences = []
    for para in doc:
        num_sentences.append(len(para))
    document_lengths.append(num_sentences)


# In[ ]:


out_dict['document_length'] = document_lengths


# In[ ]:


for key,value in out_dict.items():
    if(type(value) == list):
        print("key:{}, value_length:{}".format(key, len(value)))


# In[ ]:


lengths = set([len(s) for s in out_dict["sequence"]])
print(lengths)


# In[ ]:


lengths = set([len(s) for s in out_dict["segment_id"]])
print(lengths)


# In[ ]:


out_dict["sentence_start_index"][0]


# In[ ]:


len(question_ids)


# In[ ]:


len(out_dict["question_id"])


# In[ ]:


the_real_out_dict = {}
for i in range(max_num_paragraphs):
    the_real_out_dict["sequence_{}".format(i)] = []
    the_real_out_dict["sentence_start_index_{}".format(i)] = []
    the_real_out_dict["sentence_end_index_{}".format(i)] = []
    the_real_out_dict["supporting_fact_{}".format(i)] = []
    the_real_out_dict["segment_id_{}".format(i)] = []

the_real_out_dict["document_length"] = out_dict['document_length']

for i in trange(len(out_dict["sequence"])):
#     the_real_out_dict["sequence_{}".format(i%10)].append(out_dict["sequence"][i] + [0]*(max_num_paragraphs-1))
    the_real_out_dict["sequence_{}".format(i%10)].append(out_dict["sequence"][i])
    the_real_out_dict["sentence_start_index_{}".format(i%10)].append(out_dict["sentence_start_index"][i])
    the_real_out_dict["sentence_end_index_{}".format(i%10)].append(out_dict["sentence_end_index"][i])
    the_real_out_dict["supporting_fact_{}".format(i%10)].append(out_dict["supporting_fact"][i])
#     the_real_out_dict["segment_id_{}".format(i%10)].append(out_dict["segment_id"][i] + [1]*(max_num_paragraphs-1))
    the_real_out_dict["segment_id_{}".format(i%10)].append(out_dict["segment_id"][i])


# In[ ]:


for key,value in the_real_out_dict.items():
    if(type(value) == list):
        print("key:{}, value_length:{}".format(key, len(value)))


# In[ ]:


len(the_real_out_dict["sequence_0"][0])


# In[ ]:


len(the_real_out_dict["segment_id_0"][0])


# In[ ]:


small_out_dict = {}
for key, value in the_real_out_dict.items():
    small_out_dict[key] = value[:small_dataset_size]


# In[ ]:


for key,value in small_out_dict.items():
    if(type(value) == list):
        print("key:{}, value_length:{}".format(key, len(value)))


# In[ ]:


pickler(out_pkl_path, small_out_pkl_name, small_out_dict)
print("Done")


# In[ ]:


pickler(out_pkl_path, out_pkl_name, the_real_out_dict)
print("Done")


# In[ ]:




