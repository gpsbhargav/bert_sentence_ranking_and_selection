import os
import time
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, TensorDataset

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import numpy as np
from sklearn.metrics import f1_score, accuracy_score


import utils
from hotpot_model import SentenceSelector
import options

import pdb


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

options = options.HotpotOptions()

torch.cuda.set_device(options.gpu)
device = torch.device('cuda:{}'.format(options.gpu))

print("Reading data pickles")

train_data = utils.unpickler(options.data_pkl_path, options.train_pkl_name)
dev_data = utils.unpickler(options.data_pkl_path, options.dev_pkl_name)


train_dataset = TensorDataset(torch.tensor(train_data["sequences"]), torch.tensor(train_data["segment_ids"]),
torch.tensor(train_data["supporting_fact"], dtype=torch.float32))

dev_dataset = TensorDataset(torch.tensor(dev_data["sequences"]), torch.tensor(dev_data["segment_ids"]),
torch.tensor(dev_data["supporting_fact"], dtype=torch.float32))

train_data_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True, sampler=None, batch_sampler=None, num_workers=0, pin_memory=True, drop_last=False, timeout=0, worker_init_fn=None)

dev_data_loader = DataLoader(dev_dataset, batch_size=options.dev_batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, pin_memory=True, drop_last=False, timeout=0, worker_init_fn=None)



print("Building model")

model = SentenceSelector(options, device)

print("===============================")
print("Model:")
print(model)
print("===============================")

if torch.cuda.device_count() > 1:
  print("Using", torch.cuda.device_count(), "GPUs")
  model = nn.DataParallel(model)

model.to(device)

criterion = nn.BCEWithLogitsLoss()
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
            len(train_data["sequences"]) / options.batch_size / options.gradient_accumulation_steps * options.epochs)

t_total = num_train_steps
optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=options.learning_rate,
                             warmup=options.warmup_proportion,
                             t_total=t_total)



iterations = 0
start = time.time()
best_dev_f1 = -1


routine_log_template = 'Time:{:.2f}, Epoch:{}/{}, Iteration:{}, Avg_train_loss:{:.4f}, batch_loss:{:.4f}, batch_EM:{:.4f}, batch_F1:{:.4f}'

dev_log_template = 'Dev set - Exact match:{:.4f}, F1:{:.4f}'

print("Training data size:{}".format(len(train_dataset)))
print("Dev data size:{}".format(len(dev_dataset)))
print("Training now")

total_loss_since_last_time = 0

num_evaluations_since_last_best_dev_acc  = 0

dev_predictions_best_model = None

stop_training_flag = False

for epoch in range(options.epochs):
    
    for batch_idx, batch in enumerate(train_data_loader):
        
        batch = [t.to(device) for t in batch]

        model.train(); optimizer.zero_grad()
        
        iterations += 1
        
        answer = model(batch[0], batch[1])
        
        gt_labels = batch[2]
    
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
            thresholded_answer = torch.sigmoid(answer) > options.decision_threshold
            
            train_exact_match = accuracy_score(gt_labels.cpu().numpy(), thresholded_answer.detach().cpu().numpy())
            
            train_f1 = f1_score(gt_labels.cpu().numpy(), thresholded_answer.detach().cpu().numpy(),average='micro')
            
            avg_loss = total_loss_since_last_time/options.log_every
            total_loss_since_last_time = 0
            
            print(routine_log_template.format(time.time()-start, epoch+1, options.epochs, iterations,avg_loss, loss.item(), train_exact_match, train_f1))
        
        
            if iterations % options.save_every == 0:
                snapshot_prefix = os.path.join(options.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + '_f1_{:.4f}_loss_{:.4f}_iter_{}_model.pt'.format(train_f1, loss.item(), iterations)
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)
     
        if iterations % options.evaluate_every == 0:
            print("Evaluating on dev set")

            # switch model to evaluation mode
            model.eval()

            answers_for_whole_dev_set = []
            with torch.no_grad():
                for dev_batch_idx, dev_batch in enumerate(dev_data_loader):
                    dev_batch = [t.to(device) for t in dev_batch]
                    dev_answer = model(dev_batch[0], dev_batch[1])
                    answers_for_whole_dev_set.append(dev_answer.cpu().numpy())

            answers_for_whole_dev_set = np.concatenate(answers_for_whole_dev_set, axis = 0)

            dev_answer_labels = (torch.sigmoid(torch.tensor(answers_for_whole_dev_set)) > options.decision_threshold).numpy()
            
            dev_exact_match = accuracy_score(np.array(dev_data["supporting_fact"]), dev_answer_labels)
            
            dev_f1 = f1_score(np.array(dev_data["supporting_fact"]), dev_answer_labels, average='micro')

            print(dev_log_template.format(dev_exact_match,dev_f1))


            # update best valiation set accuracy
            if dev_f1 > best_dev_f1:
                
                dev_predictions_best_model = answers_for_whole_dev_set
                
                num_evaluations_since_last_best_dev_acc = 0
                
                # found a model with better validation set accuracy

                best_dev_f1 = dev_f1
                snapshot_prefix = os.path.join(options.save_path, 'best_snapshot')
                snapshot_path = snapshot_prefix + '_dev_f1_{}_iter_{}_model.pt'.format(dev_f1, iterations)

                # save model, delete previous 'best_snapshot' files
                torch.save(model.state_dict(), snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)
            else:
                num_evaluations_since_last_best_dev_acc += 1
            
            if(num_evaluations_since_last_best_dev_acc > options.early_stopping_patience):
                print("Training stopped because dev acc hasn't increased in {} epochs.".format(options.early_stopping_patience))
                print("Best dev set accuracy = {}".format(best_dev_f1))
                stop_training_flag = True
                break

    if(stop_training_flag == True):
        break

# save best predictions
if(dev_predictions_best_model is not None):
    utils.pickler(options.save_path, options.predictions_pkl_name, dev_predictions_best_model)

