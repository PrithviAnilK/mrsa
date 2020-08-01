import os
import sys
from config import get_config 
import torch
from models import get_model
from dataloader import get_dataloader
import pandas as pd 
from tqdm import tqdm 


def save_prediction(predictions, ids, version):    
    df = pd.DataFrame(list(zip(ids, predictions)), columns = ['PhraseId', 'Sentiment'])
    df.to_csv('{}.csv'.format(version), index = False)


def predict(net, dataloader, device):
    net.eval()
    predictions = []
    ids = []
    runner = tqdm(dataloader, total = len(dataloader))
    for x, phrase_id in runner:
        batch_size = x.size()[0]
        x.to(device)
        output = net(x)
        preds = torch.argmax(output, dim = 1)
        preds = preds[0].item()
        phrase_id = phrase_id[0].item()
        predictions.append(preds)
        ids.append(phrase_id)
    return predictions, ids

if __name__ == "__main__":
    version = sys.argv[-1] 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config = get_config(version)
    net = get_model(config)

    model_path = config["MODEL_PATH"]
    load_path = os.path.join(model_path, '{}.pth'.format(version))
    test_dataloader = get_dataloader(config, "TEST")
    net.load_state_dict(torch.load(load_path))
    predictions, ids = predict(net, test_dataloader, device)
    save_prediction(predictions, ids, version)