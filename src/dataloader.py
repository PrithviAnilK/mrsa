import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
import string
import json

class SentimentDataset(Dataset):
    def __init__(self, path, word_to_index_path, _type):
        super(SentimentDataset, self).__init__()
        self.df = pd.read_csv(path, sep = '\t')
        self.Phrase = self.df.Phrase.values
        self._type = _type
        
        if _type == "TRAIN":
            self.Sentiment = self.df.Sentiment.values
        elif _type == "TEST":
            self.PhraseId = self.df.PhraseId.values
        
        self.nlp = spacy.load('en')
        f = open(word_to_index_path, 'r')
        self.word_to_index = json.load(f)
        f.close()
    
    def __len__(self): return self.df.shape[0]

    def __getitem__(self, dex):
        phrase = self.Phrase[dex].lower()
        cleaned_phrase = [str(word.lemma_) for word in self.nlp(phrase) if word.lemma_ not in STOP_WORDS]
        X = list(map(lambda x: self.word_to_index[x] if x in self.word_to_index else 13553, cleaned_phrase))
        X += [13553] * (35 - len(X))
        X = torch.LongTensor(X)
        if self._type == "TRAIN":
            Y = self.Sentiment[dex]
            return (X, Y)
        elif self._type == "TEST":
            PhraseId = self.PhraseId[dex]
            return (X, PhraseId)


def get_dataloader(config, _type):
    path, word_to_index_path, batch_size, shuffle, num_workers = get_dataloader_configs(config, _type)
    dataset = SentimentDataset(path, word_to_index_path, _type)
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle = shuffle, 
        num_workers = num_workers
    )
    return dataloader


def get_dataloader_configs(config, _type):
    if _type == "TRAIN":
        return config["TRAIN_PATH"], config["WORD_TO_INDEX_PATH"], config["BATCH_SIZE"], config["SHUFFLE"], config["NUM_WORKERS"]
    elif _type == "TEST":
        return config["TEST_PATH"], config["WORD_TO_INDEX_PATH"],  1, False, config["NUM_WORKERS"]