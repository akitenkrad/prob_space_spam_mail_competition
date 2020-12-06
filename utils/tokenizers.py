import string
import re

def preprocessing_text(text):
    # to lower
    text = text.lower()
    
    # remove specific tokens
    text = re.sub('<br />', '', text)
    text = re.sub('subject', '', text)
    
    # remove punctuations
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    
    return text

def tokenizer_punctuation(text):
    return text.strip().split()

def tokenizer(text):
    text = preprocessing_text(text)
    words = [word.strip() for word in text.strip().split()]
    return words
