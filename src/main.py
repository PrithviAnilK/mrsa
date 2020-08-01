import sys
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
from config import get_config
from dataloader import get_dataloader 
from models import get_model
from trainer import train

def reproduce(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if str(device) == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    reproduce(23, device)
    
    version = sys.argv[-1] 
    config = get_config(version)
    net = get_model(config)
    train_dataloader = get_dataloader(config, "TRAIN")
    
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr = config["ALPHA"])
    criterion = nn.CrossEntropyLoss()
    train(net, train_dataloader, criterion, optimizer, device, config, version)