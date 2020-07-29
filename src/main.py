import torch
import torch.optim as optim
import torch.nn as nn
from config import TRAIN_PATH, NUM_WORKERS, BATCH_SIZE, SHUFFLE, WORD_TO_INDEX_PATH, MODEL_TYPE, VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BIDIRECTIONAL, ALPHA, EPOCHS, MODEL_PATH, SAVE
from dataloader import get_dataloader 
from models import SeqModel
from trainer import train

def reproduce(seed, device):
    torch.manual_seed(seed) 
    if str(device) == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":    
    device = torch.device('cuda')
    reproduce(23, device)
    
    train_dataloader = get_dataloader(TRAIN_PATH, WORD_TO_INDEX_PATH,  BATCH_SIZE, SHUFFLE, NUM_WORKERS)
    
    net = SeqModel(
        MODEL_TYPE, 
        VOCAB_SIZE, 
        EMBEDDING_SIZE, 
        HIDDEN_SIZE, 
        NUM_LAYERS, 
        DROPOUT, 
        BIDIRECTIONAL
    )   
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr = ALPHA)
    criterion = nn.CrossEntropyLoss()
    train(net, train_dataloader, EPOCHS, BATCH_SIZE, criterion, optimizer, device, MODEL_PATH, SAVE)