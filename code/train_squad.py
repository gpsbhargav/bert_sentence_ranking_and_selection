import os
import time
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import utils
from model import SentenceSelector
import options

import pdb


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

options = options.SquadOptions()

torch.cuda.set_device(options.gpu)
device = torch.device('cuda:{}'.format(options.gpu))

print("Reading data pickles")

train_data = utils.unpickler(options.data_pkl_path, options.train_pkl_name)
dev_data = utils.unpickler(options.data_pkl_path, options.dev_pkl_name)

print("Building model")

model = SentenceSelector(options, device)

model.to(device)

print("===============================")
print("Model:")
print(model)
print("===============================")


criterion = nn.CrossEntropyLoss()
# opt = BertAdam(model.parameters(), lr=options.lr,  weight_decay=options.weight_decay)


# Prepare optimizer
param_optimizer = list(model.named_parameters())

# hack to remove pooler, which is not used
# thus it produce None grad that break apex
# param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

num_train_steps = int(
            len(train_data["sequence_0"]) / options.batch_size / options.gradient_accumulation_steps * options.epochs)

t_total = num_train_steps
optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=options.learning_rate,
                             warmup=options.warmup_proportion,
                             t_total=t_total)



iterations = 0
start = time.time()
best_dev_exact_match = -1


routine_log_template = 'Time:{:.4f}, Epoch:{}/{}, Iteration:{}, Avg_train_loss:{:.4f}, batch_loss:{:.4f}, Train_EM:{:.4f}'

dev_log_template = 'Dev set - Exact match:{}'



train_data_len = len(train_data["sequence_0"])
dev_data_len = len(dev_data["sequence_0"])

train_minibatch_from_indices = utils.MinibatchFromIndices(train_data, device)
dev_minibatch_from_indices = utils.MinibatchFromIndices(dev_data, device)

del train_data

print("Training now")

total_loss_since_last_time = 0

num_epochs_since_last_best_dev_acc  = 0

dev_predictions_best_model = None

stop_training_flag = False

for epoch in range(options.epochs):
    
    train_minibatch_index_generator = utils.minibatch_indices_generator(train_data_len, options.batch_size)
    
    
    for batch_idx, minibatch_indices in enumerate(train_minibatch_index_generator):
        batch = train_minibatch_from_indices.get(minibatch_indices)
        
        model.train(); optimizer.zero_grad()
        
        iterations += 1
        
        answer = model(batch)
        
        gt_labels = batch["supporting_fact"].int().argmax(dim=1)
        
        loss = criterion(answer, gt_labels) 
        
        if(torch.isnan(loss).item()):
            print("Loss became nan in iteration {}. Training stopped".format(iterations))
            stop_training_flag = True
            break
        elif(loss.item() < 0.0000000000001):
            print("Loss is too low. Stopping training")
            stop_training_flag = True
            break
        
        total_loss_since_last_time += loss.item()
        loss.backward()
        
        lr_this_step = options.learning_rate * warmup_linear(iterations/t_total, options.warmup_proportion)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_step
        optimizer.step()
        
        
        
        if iterations % options.log_every == 0:
            answer_copy = answer
            train_exact_match = accuracy_score(gt_labels.cpu().numpy(), answer_copy.detach().cpu().numpy().argmax(axis=1))
            
            avg_loss = total_loss_since_last_time/options.log_every
            total_loss_since_last_time = 0
            
            print(routine_log_template.format(time.time()-start, epoch+1, options.epochs, iterations,avg_loss, loss.item(), train_exact_match))
        
        
            if iterations % options.save_every == 0:
                snapshot_prefix = os.path.join(options.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + '_EM_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_exact_match, loss.item(), iterations)
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)
     
    if(stop_training_flag == True):
        break
                    
    print("Evaluating on dev set")
    dev_minibatch_index_generator = utils.minibatch_indices_generator(dev_data_len, options.batch_size, shuffle=False)

    # switch model to evaluation mode
    model.eval();

    answers_for_whole_dev_set = []
    total_dev_loss = 0
    with torch.no_grad():
        for dev_batch_idx, dev_minibatch_indices in enumerate(dev_minibatch_index_generator):
            dev_batch = dev_minibatch_from_indices.get(dev_minibatch_indices)
            dev_answer = model(dev_batch)
            answers_for_whole_dev_set.append(dev_answer.cpu().numpy())
            dev_gt_labels = dev_batch["supporting_fact"].int().argmax(dim=1)
            dev_loss = criterion(dev_answer, dev_gt_labels)
            total_dev_loss += dev_loss.item()

    answers_for_whole_dev_set = np.concatenate(answers_for_whole_dev_set, axis = 0)

    dev_answer_labels = answers_for_whole_dev_set.argmax(axis=1)
    
    dev_exact_match = accuracy_score(np.array(dev_data["supporting_fact"]).argmax(axis = 1), dev_answer_labels)

    print(dev_log_template.format(dev_exact_match))


    # update best valiation set accuracy
    if dev_exact_match > best_dev_exact_match:
        
        dev_predictions_best_model = answers_for_whole_dev_set
        
        num_epochs_since_last_best_dev_acc = 0
        
        # found a model with better validation set accuracy

        best_dev_exact_match = dev_exact_match
        snapshot_prefix = os.path.join(options.save_path, 'best_snapshot')
        snapshot_path = snapshot_prefix + '_dev_EM_{}_iter_{}_model.pt'.format(dev_exact_match, iterations)

        # save model, delete previous 'best_snapshot' files
        torch.save(model.state_dict(), snapshot_path)
        for f in glob.glob(snapshot_prefix + '*'):
            if f != snapshot_path:
                os.remove(f)
    else:
        num_epochs_since_last_best_dev_acc += 1
    
    if(num_epochs_since_last_best_dev_acc > options.early_stopping_patience):
        print("Training stopped because dev acc hasn't increased in {} epochs.".format(options.early_stopping_patience))
        print("Best dev set accuracy = {}".format(best_dev_exact_match))
        break

        

# save best predictions
if(dev_predictions_best_model is not None):
    utils.pickler(options.save_path, options.predictions_pkl_name, dev_predictions_best_model)
