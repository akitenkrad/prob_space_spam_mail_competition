from pathlib import Path
from tqdm import tqdm
import numpy as np
from utils.vocabulary import Vocabulary

def load_fasttext_weights(weights, word_vocab:Vocabulary):
    weights = Path(weights)
    
    with open(weights) as f:
        total, dim = map(int, f.readline().split())
        vocab_size = len(word_vocab.tokens)
        weight_matrix = np.zeros((vocab_size, dim))
        
        for line in tqdm(f, desc='Loading fasttext weights', total=total):
            tokens = line.strip().split()
            token = tokens[0]
            idx = word_vocab.token2idx(token)
            weight_matrix[idx] = [float(p) for p in tokens[1:]]
            
    return weight_matrix
