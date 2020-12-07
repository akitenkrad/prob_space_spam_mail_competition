from pathlib import Path
import math
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchtext
from utils.fasttext import load_fasttext_weights

class Embedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        x_vec = self.embeddings(x)

        return x_vec
    
class PositionalEncoder(nn.Module):
    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))

        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, x):
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret
    
class Attention(nn.Module):
    def __init__(self, d_model=300):
        super().__init__()
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)
        self.d_k = d_model

    def forward(self, q, k, v, mask):
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)
        normalized_weights = F.softmax(weights, dim=-1)
        output = torch.matmul(normalized_weights, v)
        output = self.out(output)

        return output, normalized_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_normalized = self.norm_1(x)
        output, normalized_weights = self.attn(x_normalized, x_normalized, x_normalized, mask)
        x2 = x + self.dropout_1(output)
        x_normalized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normalized2))

        return output, normalized_weights
    
class ClassificationHead(nn.Module):
    def __init__(self, d_model=300, output_dim=2):
        super().__init__()
        self.linear = nn.Linear(d_model, output_dim)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        x0 = x[:, 0, :]
        out = self.linear(x0)
        return out
    
class TransformerClassification(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=300, d_model=300, max_seq_len=256, output_dim=1):
        super().__init__()
        self.net1 = Embedder(vocab_size, embedding_dim)
        self.net2 = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.net3_1 = TransformerBlock(d_model=d_model)
        self.net3_2 = TransformerBlock(d_model=d_model)
        self.net4 = ClassificationHead(output_dim=output_dim, d_model=d_model)
        self.sigmoid = nn.Sigmoid()

    def load_fasttext_embedding(self, weights, word_vocab):
        weight_cache = Path(__file__).parent.parent / '__cache__' / 'fasttext_weight_matrix.pickle'
        weight_cache.parent.mkdir(parents=True, exist_ok=True)
        if weight_cache.exists():
            weight_matrix = pickle.load(open(weight_cache, 'rb'))
            print('Loaded fasttext weights from cache:', str(weight_cache))
        else:
            weight_matrix = load_fasttext_weights(weights, word_vocab)
            pickle.dump(weight_matrix, open(weight_cache, 'wb'))
        self.net1.embeddings.weight.data.copy_(torch.tensor(weight_matrix))
 
    def forward(self, x, mask):
        x1 = self.net1(x)
        x2 = self.net2(x1)
        x3_1, normalized_weights_1 = self.net3_1(x2, mask)
        x3_2, normalized_weights_2 = self.net3_2(x3_1, mask)
        x4 = self.net4(x3_2)
        x4 = self.sigmoid(x4)
        return x4, normalized_weights_1, normalized_weights_2
