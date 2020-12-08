from pathlib import Path
import yaml
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
def weights_init(layer):
    classname = layer.__class__.__name__
    if classname.find('Linear') != -1:
        # initialize Linear layer
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)

def read_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def show_train_settings(config):
    fmt = '{:25s}: {}'
    print()
    print('<<< settings >>>')
    print('='*50)
    print(fmt.format('Epochs', config['config']['epochs']))
    print(fmt.format('Batch Size', config['config']['batch_size']))
    print(fmt.format('Model', config['config']['model']))
    print(fmt.format('Data Path', config['config']['data_path']))
    print(fmt.format('Max Sentence Length', config['config']['max_sent_len']))
    print(fmt.format('Validation Dataset Size', config['config']['valid_size']))
    print(fmt.format('Learning Rate', config['config']['learning_rate']))
    print(fmt.format('Save Directory', config['config']['save_dir']))
    print(fmt.format('Log Directory', config['config']['log_dir']))
    print('='*50)
    print()

def show_test_settings(config):
    fmt = '{:25s}: {}'
    print()
    print('<<< settings >>>')
    print('='*50)
    print(fmt.format('Batch Size', config['config']['batch_size']))
    print(fmt.format('Model', config['config']['model']))
    print(fmt.format('Data Path', config['config']['data_path']))
    print(fmt.format('Max Sentence Length', config['config']['max_sent_len']))
    print(fmt.format('Weights', config['config']['weights_path']))
    print(fmt.format('Output Path', str(Path(config['config']['out_dir']) / 'predicts.csv')))
    print('='*50)
    print()
   
class History(object):
    def __init__(self, log_dir):
        self.history = SummaryWriter(log_dir=log_dir)
        self.best_f1 = -1.0
        self.is_best = False
        self.epoch = 0
        self.loss = -1
        self.train_accuracy, self.train_precision, self.train_recall, self.train_f1 = 0.0, 0.0, 0.0, 0.0
        self.train_TN, self.train_FP, self.train_FN, self.train_TP = 0, 0, 0, 0
        self.test_accuracy, self.test_precision, self.test_recall, self.test_f1 = 0.0, 0.0, 0.0, 0.0
        self.test_TN, self.test_FP, self.test_FN, self.test_TP = 0, 0, 0, 0
        
    def add_train_value(self, epoch, outputs, ys, loss):
        self.epoch = epoch
        self.loss = loss
        self.train_accuracy = accuracy_score(ys, outputs)
        self.train_precision = precision_score(ys, outputs)
        self.train_recall = recall_score(ys, outputs)
        self.train_f1 = f1_score(ys, outputs)
        self.train_TN, self.train_FP, self.train_FN, self.train_TP = confusion_matrix(ys, outputs).flatten()
        
        self.history.add_scalar('loss', self.loss, epoch)
        self.history.add_scalar('train_accuracy', self.train_accuracy, epoch)
        self.history.add_scalar('train_precision', self.train_precision, epoch)
        self.history.add_scalar('train_recall', self.train_recall, epoch)
        self.history.add_scalar('train_f1', self.train_f1, epoch)
        self.history.add_scalar('train_TN', self.train_TN, epoch)
        self.history.add_scalar('train_FP', self.train_FP, epoch)
        self.history.add_scalar('train_FN', self.train_FN, epoch)
        self.history.add_scalar('train_TP', self.train_TP, epoch)
        
        if self.best_f1 < self.train_f1:
            self.best_f1 = self.train_f1
            self.is_best = True
        else:
            self.is_best = False
        
    def add_test_value(self, epoch, outputs, ys):
        self.test_accuracy = accuracy_score(ys, outputs)
        self.test_precision = precision_score(ys, outputs)
        self.test_recall = recall_score(ys, outputs)
        self.test_f1 = f1_score(ys, outputs)
        self.test_TN, self.test_FP, self.test_FN, self.test_TP = confusion_matrix(ys, outputs).flatten()
        
        self.history.add_scalar('test_accuracy', self.test_accuracy, epoch)
        self.history.add_scalar('test_precision', self.test_precision, epoch)
        self.history.add_scalar('test_recall', self.test_recall, epoch)
        self.history.add_scalar('test_f1', self.test_f1, epoch)
        self.history.add_scalar('test_TN', self.test_TN, epoch)
        self.history.add_scalar('test_FP', self.test_FP, epoch)
        self.history.add_scalar('test_FN', self.test_FN, epoch)
        self.history.add_scalar('test_TP', self.test_TP, epoch)
    
    def description(self):
        desc = 'Epoch:{} Loss:{:.3f} Tran-F1:{:.3f} Test-F1:{:.3f}'.format(self.epoch, self.loss, self.train_f1, self.test_f1)
        return desc
    