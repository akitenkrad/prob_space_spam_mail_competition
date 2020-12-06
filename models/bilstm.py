import torch
import torch.nn as nn
from utils.fasttext import load_fasttext_weights

class BiLSTM(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=300, hidden_size=150):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.linear_1 = nn.Linear(hidden_size*2, 16)
        self.dropout = nn.Dropout(0.1)
        self.linear_2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def load_fasttext_embedding(self, weights, word_vocab):
        weight_matrix = load_fasttext_weights(weights, word_vocab)
        self.embedding.weight.data.copy_(torch.tensor(weight_matrix))
        for p in self.embedding.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        '''
        Args:
            x: (batch_size, seq_len)
        '''
        embed = self.embedding(x)
        _, hidden = self.gru(embed)
        out = self.linear_1(hidden.reshape(-1, self.hidden_size*2))
        out = self.dropout(out)
        out = self.linear_2(out)
        out = self.sigmoid(out)
        return out
    