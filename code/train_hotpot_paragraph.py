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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


import utils
from model_hotpot_paragraph import SentenceSelector
import options

import pdb


def evaluate(gt, pred):
    assert(len(gt) == len(pred))
    total_size = len(pred)
    assert(len(gt) != 0)
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_correct = 0
    for i in range(total_size):
        if(np.array_equal(gt[i], pred[i])):
            total_correct += 1
        p = precision_score(gt[i], pred[i],average="binary")
        r = recall_score(gt[i], pred[i],average="binary")
        total_precision += p
        total_recall += r
        total_f1 += 2*(p*r)/(p+r) if (p+r)>0 else 0
    return {"precision":total_precision/total_size, "recall":total_recall/total_size, 
            "f1":total_f1/total_size, "em":total_correct/total_size}


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


class HotpotParagraphDataset(Dataset):

    def __init__(self, data):
        self.data = data
        self.max_seq_len = data['max_seq_len']
        self.max_sentences = data['max_sentences']

    def __len__(self):
        return len(self.data['sequence'])

    def __getitem__(self, index):
        sentence_start_indices = self.data['sentence_start_index'][index]
        sentence_end_indices = self.data['sentence_end_index'][index]
        sequence = self.data['sequence'][index]
        segment_id = self.data['segment_id'][index]
        supporting_fact = self.data['supporting_fact'][index]

        start_index_matrix = []
        end_index_matrix = []
        for i in range(len(sentence_start_indices)):
            start_indicator_vector = [0] * self.max_seq_len
            end_indicator_vector = [0] * self.max_seq_len
            start_indicator_vector[sentence_start_indices[i]] = 1
            end_indicator_vector[sentence_end_indices[i]] = 1
            start_index_matrix.append(start_indicator_vector)
            end_index_matrix.append(end_indicator_vector)

        if(self.max_sentences - len(sentence_start_indices) > 0):
            fake_sentence_start_and_end = [0] * self.max_seq_len
            index_of_pad = sequence.index(0) #pad_index is 0
            fake_sentence_start_and_end[index_of_pad] = 1
            for i in range(self.max_sentences - len(sentence_start_indices)):
                start_index_matrix.append(fake_sentence_start_and_end)
                end_index_matrix.append(fake_sentence_start_and_end)
        
        assert(len(start_index_matrix) == self.max_sentences)
        assert(len(end_index_matrix) == self.max_sentences)

        return torch.tensor(sequence), torch.tensor(segment_id), torch.tensor(start_index_matrix, dtype=torch.float32), torch.tensor(end_index_matrix, dtype=torch.float32), torch.tensor(supporting_fact, dtype=torch.float32)


options = options.ParagraphHotpotOptions()

torch.cuda.set_device(options.gpu)
device = torch.device('cuda:{}'.format(options.gpu))

print("Reading data pickles")

train_data = utils.unpickler(options.data_pkl_path, options.train_pkl_name)
dev_data = utils.unpickler(options.data_pkl_path, options.dev_pkl_name)

train_dataset = HotpotParagraphDataset(train_data)

dev_dataset = HotpotParagraphDataset(dev_data)

train_data_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True, sampler=None, batch_sampler=None, num_workers=8, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)

dev_data_loader = DataLoader(dev_dataset, batch_size=options.dev_batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=8, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)


print("Building model")

model = SentenceSelector(options, device)

print("===============================")
print("Model:")
print(model)
print("===============================")

if torch.cuda.device_count() > 1 and options.use_multiple_gpu:
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
            len(train_data["sequence"]) / options.batch_size / options.gradient_accumulation_steps * options.epochs)

t_total = num_train_steps
optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=options.learning_rate,
                             warmup=options.warmup_proportion,
                             t_total=t_total)

routine_log_template = 'Time:{:.2f}, Epoch:{}/{}, Iteration:{}, Avg_train_loss:{:.4f}, batch_loss:{:.4f}, batch_EM:{:.4f}, batch_F1:{:.4f}'

dev_log_template = 'Dev set - Exact match:{:.4f}, F1:{:.4f}'

print("Training data size:{}".format(len(train_dataset)))
print("Dev data size:{}".format(len(dev_dataset)))
print("Training now")

total_loss_since_last_time = 0
num_evaluations_since_last_best_dev_acc  = 0
dev_predictions_best_model = None
stop_training_flag = False

iterations = 0
best_dev_f1 = -1
start_epoch = 0


if options.resume_training:
    if os.path.isfile(os.path.join(options.save_path, options.checkpoint_name)):
        print("=> loading checkpoint")
        checkpoint = torch.load(os.path.join(options.save_path, options.checkpoint_name))
        start_epoch = checkpoint['epoch']
        best_dev_f1 = checkpoint['best_acc']
        iterations = checkpoint['iteration']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint. Resuming epoch {}, iteration {}"
              .format(checkpoint['epoch']+1, checkpoint['iteration']))
        

start = time.time()

for epoch in range(start_epoch, options.epochs):
    
    for batch_idx, batch in enumerate(train_data_loader):
        
        batch = [t.to(device) for t in batch]

        model.train(); optimizer.zero_grad()
        
        iterations += 1

        answer = model(batch[0], batch[1], batch[2], batch[3])
        
        gt_labels = batch[4]
    
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
            
            metrics = evaluate(gt_labels.cpu().numpy(), thresholded_answer.detach().cpu().numpy())

            train_exact_match = metrics["em"]
            
            train_f1 = metrics["f1"]
            
            avg_loss = total_loss_since_last_time/options.log_every
            total_loss_since_last_time = 0
            
            print(routine_log_template.format(time.time()-start, epoch+1, options.epochs, iterations,avg_loss, loss.item(), train_exact_match, train_f1))
            print("Number of 1s in GT:{}, Number of 1s in prediction:{}".format(gt_labels.cpu().numpy().sum(), thresholded_answer.detach().cpu().numpy().sum()))
        
        
            if iterations % options.save_every == 0:
                snapshot_prefix = os.path.join(options.save_path, options.checkpoint_name)
                snapshot_path = snapshot_prefix
                state = {
                            'epoch': epoch,
                            'iteration': iterations,
                            'state_dict': model.state_dict(),
                            'best_acc': best_dev_f1,
                            'optimizer' : optimizer.state_dict()
                        }
                torch.save(state, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)
     
    if(stop_training_flag == True):
        break
    
    print("Evaluating on dev set")

    # switch model to evaluation mode
    model.eval()

    answers_for_whole_dev_set = []
    with torch.no_grad():
        for dev_batch_idx, dev_batch in enumerate(dev_data_loader):
            dev_batch = [t.to(device) for t in dev_batch]
            dev_answer = model(dev_batch[0], dev_batch[1], dev_batch[2], dev_batch[3])
            answers_for_whole_dev_set.append(dev_answer.cpu().numpy())

    answers_for_whole_dev_set = np.concatenate(answers_for_whole_dev_set, axis = 0)

    dev_answer_labels = (torch.sigmoid(torch.tensor(answers_for_whole_dev_set)) > options.decision_threshold).numpy()
    
    dev_metrics = evaluate(np.array(dev_data["supporting_fact"]), dev_answer_labels)

    dev_exact_match = dev_metrics["em"]
    
    dev_f1 = dev_metrics["f1"]

    print(dev_log_template.format(dev_exact_match,dev_f1))
    # print("Number of 1s in GT:{}, Number of 1s in prediction:{}".format(sum(dev_data["supporting_fact"]), dev_answer_labels.sum()))


    # update best valiation set accuracy
    if dev_f1 > best_dev_f1:
        
        dev_predictions_best_model = answers_for_whole_dev_set
        
        num_evaluations_since_last_best_dev_acc = 0
        
        # found a model with better validation set accuracy

        best_dev_f1 = dev_f1
        snapshot_prefix = os.path.join(options.save_path, 'best_snapshot')
        snapshot_path = snapshot_prefix + '_dev_f1_{}_iter_{}_model.pt'.format(dev_f1, iterations)

        # save model, delete previous 'best_snapshot' files
        state = {
                    'epoch': epoch,
                    'iteration': iterations,
                    'state_dict': model.state_dict(),
                    'best_acc': best_dev_f1,
                    'optimizer' : optimizer.state_dict()
                }
        torch.save(state, snapshot_path)
        for f in glob.glob(snapshot_prefix + '*'):
            if f != snapshot_path:
                os.remove(f)
        
        # save best predictions
        utils.pickler(options.save_path, options.predictions_pkl_name, dev_predictions_best_model)
    else:
        num_evaluations_since_last_best_dev_acc += 1
    
    if(num_evaluations_since_last_best_dev_acc > options.early_stopping_patience):
        print("Training stopped because dev acc hasn't increased in {} epochs.".format(options.early_stopping_patience))
        print("Best dev set accuracy = {}".format(best_dev_f1))

    
    

