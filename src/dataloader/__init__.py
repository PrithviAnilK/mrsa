import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
import string
import json

class SentimentDataset(Dataset):
    def __init__(self, path, word_to_index_path):
        super(SentimentDataset, self).__init__()
        self.df = pd.read_csv(path, sep = '\t')
        self.Phrase = self.df.Phrase.values
        self.Sentiment = self.df.Sentiment.values
        self.nlp = spacy.load('en')
        f = open(word_to_index_path, 'r')
        self.word_to_index = json.load(f)
        f.close()
    
    def __len__(self): return self.df.shape[0]

    def __getitem__(self, dex):
        phrase = self.Phrase[dex].lower()
        cleaned_phrase = [str(word.lemma_) for word in self.nlp(phrase) if word.lemma_ not in STOP_WORDS]
        X = list(map(lambda x: self.word_to_index[x], cleaned_phrase))
        X += [13553] * (35 - len(X))
        Y = self.Sentiment[dex]
        X = torch.LongTensor(X)
        return (X, Y)

def get_dataloader(path, word_to_index_path, batch_size, shuffle, num_workers):
    dataset = SentimentDataset(path, word_to_index_path)
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle = shuffle, 
        num_workers = num_workers
    )
    return dataloader