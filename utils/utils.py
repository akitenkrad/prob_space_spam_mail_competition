import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

class History(object):
    def __init__(self, log_dir):
        self.history = SummaryWriter(log_dir=log_dir)
        self.best_accuracy = -1.0
        self.is_best = False
        self.epoch = 0
        self.loss = -1
        self.train_accuracy, self.train_precision, self.train_recall, self.train_f1 = 0.0, 0.0, 0.0, 0.0
        self.test_accuracy, self.test_precision, self.test_recall, self.test_f1 = 0.0, 0.0, 0.0, 0.0
    
    def add_train_value(self, epoch, outputs, ys, loss):
        self.epoch = epoch
        self.loss = loss
        self.train_accuracy = accuracy_score(ys, outputs)
        self.train_precision = precision_score(ys, outputs)
        self.train_recall = recall_score(ys, outputs)
        
        self.history.add_scalar('loss', self.loss, epoch)
        self.history.add_scalar('train_accuracy', self.train_accuracy, epoch)
        self.history.add_scalar('train_precision', self.train_precision, epoch)
        self.history.add_scalar('train_recall', self.train_recall, epoch)
        
        if self.best_accuracy < self.train_accuracy:
            self.best_accuracy = self.train_accuracy
            self.is_best = True
        else:
            self.is_best = False
        
    def add_test_value(self, epoch, outputs, ys):
        self.test_accuracy = accuracy_score(ys, outputs)
        self.test_precision = precision_score(ys, outputs)
        self.test_recall = recall_score(ys, outputs)
        
        self.history.add_scalar('test_accuracy', self.test_accuracy, epoch)
        self.history.add_scalar('test_precision', self.test_precision, epoch)
        self.history.add_scalar('test_recall', self.test_recall, epoch)
    
    def description(self):
        desc = 'Epoch:{} Loss:{:.6f} Tran-Acc:{:.6f} Test-Acc:{:.6f}'.format(self.epoch, self.loss, self.train_accuracy, self.test_accuracy)
        return desc
    