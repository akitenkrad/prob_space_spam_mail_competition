from collections import Counter

class Vocabulary(object):
    
    def __init__(self, pad='<pad>', unk='<unk>', sos='<sos>', eos='<eos>'):
        self.__token2idx = {}
        self.__idx2token = {}
        self.counter = Counter()
        self.PAD = pad
        self.UNK = unk
        self.SOS = sos
        self.EOS = eos
           
    def __len__(self):
        return len(self.__token2idx)
    
    @property
    def tokens(self):
        return list(self.__token2idx.keys())
    
    @property
    def vocab_size(self):
        return len(self.tokens)
    
    @property
    def __next_token_idx(self):
        if len(self.__idx2token) < 1:
            return 0
        return max(self.__idx2token) + 1
    
    def add_special_tokens(self):
        self.add_token(self.UNK)
        self.add_token(self.PAD)
        self.add_token(self.SOS)
        self.add_token(self.EOS)
        
    def add_token(self, token):
        '''used to add a token from corpus'''
        if token not in self.__token2idx:
            new_idx = self.__next_token_idx
            self.__token2idx[token] = new_idx
            self.__idx2token[new_idx] = token
        self.counter.update(token)
    
    def add_token_with_idx(self, token:str, idx:int):
        '''used to import an external vocabulary such as GloVe'''
        self.__token2idx[token] = idx
        self.__idx2token[idx] = token
    
    def reset_index(self):
        '''reset token index'''
        tokens = [(token, idx) for token, idx in self.__token2idx.items()]
        tokens = sorted(tokens, key=lambda x: x[1])
        self.__token2idx, self.__idx2token = {}, {}
        for new_idx, (token, idx) in enumerate(tokens):
            self.__token2idx[token] = new_idx
            self.__idx2token[new_idx] = token
        
    def token2idx(self, token:str):
        if token not in self.__token2idx:
            if self.UNK not in self.__token2idx:
                self.add_special_tokens()
            return self.__token2idx[self.UNK]
        else:
            return self.__token2idx[token]
    
    def idx2token(self, idx:int):
        return self.__idx2token[idx]
    
    def copy(self):
        new_vocab = Vocabulary()
        for token, idx in self.__token2idx.items():
            new_vocab.add_token_with_idx(token, idx)
        return new_vocab
 