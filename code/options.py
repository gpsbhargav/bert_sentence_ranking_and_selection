class SquadOptions:
    def __init__(self):
        
        # ----Training----
        self.epochs = 3
        self.batch_size = 16
        self.log_every = 100
        self.save_every = self.log_every * 500
        self.early_stopping_patience = 4
        self.gradient_accumulation_steps = 1  # only 1 is supported
        self.gpu = 0
        
        # ----Vocab and embedding----
        
        
        # ----Data sizes, sequence lengths----
        self.max_para_len = 10
        
        # ----Data location, other paths----
        self.data_pkl_path = "../data/squad/"
        self.train_pkl_name = "preprocessed_train.pkl"
        self.dev_pkl_name = "preprocessed_dev.pkl"
        self.save_path = "../saved_models/squad"
        self.predictions_pkl_name = "predictions.pkl"
        self.bert_archive = "/local_scratch/csagps/bert_archive/"
        
        # ----Network hyperparameters----
        self.bert_type = 'bert-base-uncased' # one of bert-base-uncased or bert-large-uncased
        if(self.bert_type == 'bert-base-uncased'):
            self.bert_hidden_size = 768
        else:
            self.bert_hidden_size = 1024
        
        self.dropout = 0.1  # doesn't apply to BERT
        self.learning_rate = 3e-5
        self.warmup_proportion = 0.1
        

        
class HotpotOptions:
    def __init__(self):
        
        # ----Training----
        self.epochs = 10
        self.batch_size = 128
        self.dev_batch_size = 1024
        self.log_every = 100
        self.evaluate_every = 1000
        self.save_every = self.log_every * 500
        self.early_stopping_patience = 4
        self.gradient_accumulation_steps = 1  # only 1 is supported
        self.gpu = 0 # this will be the primary GPU if there are > 1 GPU
        
        # ----Vocab and embedding----
        
        
        # ----Data sizes, sequence lengths----
        self.max_para_len = 65
        
        # ----Data location, other paths----
        self.data_pkl_path = "../data/hotpot/"
        self.train_pkl_name = "preprocessed_train.pkl"
        self.dev_pkl_name = "preprocessed_dev.pkl"
        self.save_path = "../saved_models/hotpot/"
        self.predictions_pkl_name = "predictions.pkl"
        self.bert_archive = "../../bert_archive/"
        
        # ----Network hyperparameters----
        self.bert_type = 'bert-base-uncased' # one of bert-base-uncased or bert-large-uncased
        if(self.bert_type == 'bert-base-uncased'):
            self.bert_hidden_size = 768
        else:
            self.bert_hidden_size = 1024
        
        self.dropout = 0.1  # doesn't apply to BERT
        self.learning_rate = 3e-5
        self.warmup_proportion = 0.1
        self.decision_threshold = 0.5


class ParagraphHotpotOptions:
    def __init__(self):
        
        # ----Training----
        self.epochs = 10
        self.batch_size = 32
        self.dev_batch_size = 32
        self.log_every = 100
        self.save_every = self.log_every * 5
        self.early_stopping_patience = 4
        self.gradient_accumulation_steps = 1  # only 1 is supported
        self.gpu = 0 # this will be the primary GPU if there are > 1 GPU
        self.use_multiple_gpu = True
        self.resume_training = True
        # ----Vocab and embedding----
        
        
        # ----Data sizes, sequence lengths----
        self.max_seq_len = 510
        self.max_sentences = 10
        
        # ----Data location, other paths----
        self.data_pkl_path = "../data/hotpot_doc_level/"
        self.train_pkl_name = "preprocessed_train.pkl"
        self.dev_pkl_name = "preprocessed_dev.pkl"
        self.save_path = "../saved_models/hotpot_doc_level/"
        self.predictions_pkl_name = "predictions.pkl"
        self.bert_archive = "../../bert_archive/"
        self.checkpoint_name = "snapshot.pt"
        
        # ----Network hyperparameters----
        self.bert_type = 'bert-base-uncased' # one of bert-base-uncased or bert-large-uncased
        if(self.bert_type == 'bert-base-uncased'):
            self.bert_hidden_size = 768
        else:
            self.bert_hidden_size = 1024
        
        self.dropout = 0.1  # doesn't apply to BERT
        self.learning_rate = 3e-5
        self.warmup_proportion = 0.1
        self.decision_threshold = 0.5
        

