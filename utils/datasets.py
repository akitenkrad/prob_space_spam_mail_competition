import sys
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import numpy as np
import urllib.request as request
import zipfile
import csv
import re
import pickle
import torch
from torch.utils.data import Dataset
from utils.tokenizers import tokenizer
from utils.vocabulary import Vocabulary

csv.field_size_limit(sys.maxsize)

def collate_fn(batch):
    idx, tokens, y = [torch.tensor(item) for item in list(zip(*batch))]
    return idx, tokens, y
   
class SpamDataset(Dataset):
    
    def __init__(self, data_dir, max_length=256, phase='train', padding=True, use_pretrained=True, no_cache=False):
        super().__init__()
        assert phase in ['dev', 'train', 'test']
        
        self.data_dir = Path(data_dir)
        self.data_path = self.data_dir / f'{phase}_data.csv'
        self.max_length = max_length
        self.phase = phase
        self.padding = padding
        self.use_pretrained = use_pretrained
        self.no_cache = no_cache
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        assert self.data_path.exists(), '{} does not exist.'.format(str(self.data_path))
        
        # read data
        self.data = []
        self.__read_data()
        
        # prepare fasttext
        self.fasttext_path = Path(__file__).parent / 'weights' / 'fasttext' / 'wiki-news-300d-1M.vec'
        self.fasttext_vocab_size = 0
        self.fasttext_dim = 0
        if self.use_pretrained == True and self.fasttext_path.exists() == False:
            self.__download_fasttext()
        
        # build vocabulary
        self.word_vocab = Vocabulary()
        self.__build_vocab()
        
    @property
    def data_path(self):
        return self.data_path if self.phase == 'train' else self.test_data_path
    
    def __read_data(self):
        reader = list(csv.reader([line.replace('\0', '') for line in open(self.data_path)]))
        # skip header
        reader = reader[1:]
        
        check_set = set()
        for line in tqdm(reader, desc='Reading {}'.format(self.data_path)):
            idx = int(line[0])
            content = line[1].strip()
            y = int(line[2]) if self.phase == 'train' else -1
            
            if content in check_set:
                continue
            
            self.data.append({'id': idx, 'content': content, 'y': y})
            check_set.add(content)
        
    def __download_fasttext(self):
        print('Downloading fasttext model...')
        self.fasttext_path.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip'
        save_path = self.fasttext_path.parent / (self.fasttext_path.name + '.zip')
        request.urlretrieve(url, str(save_path))
        with zipfile.ZipFile(str(save_path)) as z:
            z.extractall(str(save_path.parent))
        
    def __load_fasttext_vocabulary(self):
        reader = open(self.fasttext_path)
        total, dim = map(int, reader.readline().split())
        self.fasttext_vocab_size = total
        self.fasttext_dim = dim
        for line in tqdm(reader, desc='Loading fasttext vocabulary', total=total):
            tokens = line.strip().split(' ')
            self.word_vocab.add_token(tokens[0])
        
    def __build_vocab(self):
        vocab_cache = Path(__file__).parent / '__cache__' / 'word_vocab.pickle'
        # check cache
        if vocab_cache.exists() and self.no_cache == False:
            pickle_data = pickle.load(open(str(vocab_cache), 'rb'))
            self.word_vocab = pickle_data['word_vocab']
            self.fasttext_vocab_size = pickle_data['fasttext_vocab_size']
            self.fasttext_dim = pickle_data['fasttext_dim']
            return
        else:
            vocab_cache.parent.mkdir(parents=True, exist_ok=True)
            
        # add special tokens
        self.word_vocab.add_special_tokens()
    
        # load fasttext vocabulary
        if self.use_pretrained:
            self.__load_fasttext_vocabulary()
            
        # create vocabulary
        for item in tqdm(self.data, desc='Creating vocabulary'):
            text = item['content']
            words = tokenizer(text)
            for word in words:
                self.word_vocab.add_token(word)
        
        # save cache
        pickle.dump(
            {
                'word_vocab': self.word_vocab,
                'fasttext_vocab_size': self.fasttext_vocab_size,
                'fasttext_dim': self.fasttext_dim,
            },
            open(str(vocab_cache), 'wb'))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        idx = item['id']
        tokens = [self.word_vocab.token2idx(token) for token in tokenizer(item['content'])]
        y = item['y']
        
        if self.padding:
            tokens = tokens + [self.word_vocab.token2idx(self.word_vocab.PAD)] * self.max_length
            tokens = tokens[:self.max_length]
            
        return (idx, tokens, y)
