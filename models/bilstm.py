from pathlib import Path
import pickle
import torch
import torch.nn as nn
from utils.fasttext import load_fasttext_weights
from utils.utils import weights_init

class BiLSTM(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=300, hidden_size=150, clip_value=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.linear_1 = nn.Linear(hidden_size*2, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
        for p in list(self.gru.parameters()) + list(self.linear_1.parameters()) + list(self.linear_2.parameters()):
            p.register_hook(lambda grad: grad.data.clamp_(-clip_value, clip_value))
            
        weights_init(self.linear_1)
        weights_init(self.linear_2)
        
    def load_fasttext_embedding(self, weights, word_vocab):
        weight_cache = Path(__file__).parent.parent / '__cache__' / 'fasttext_weight_matrix.pickle'
        weight_cache.parent.mkdir(parents=True, exist_ok=True)
        if weight_cache.exists():
            weight_matrix = pickle.load(open(weight_cache, 'rb'))
            print('Loaded fasttext weights from cache:', str(weight_cache))
        else:
            weight_matrix = load_fasttext_weights(weights, word_vocab)
            pickle.dump(weight_matrix, open(weight_cache, 'wb'))
        self.embedding.weight.data.copy_(torch.tensor(weight_matrix))
        
    def forward(self, x):
        '''
        Args:
            x: (batch_size, seq_len)
        '''
        embed = self.embedding(x)
        _, hidden = self.gru(embed)
        out = self.linear_1(hidden.reshape(-1, self.hidden_size*2))
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        out = self.sigmoid(out)
        return out
    