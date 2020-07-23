import torch
from torch.utils.data import Dataset, DataLoader

class SentimentDataset(Dataset):
    def __init__(self):
        super(SentimentDataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    
def get_dataloader(path, shuffle, batch_size):
    pass