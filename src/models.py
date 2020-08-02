import torch 
import torch.nn as nn  
import torch.nn.functional as F 
import numpy as np

class SeqModel(nn.Module):
    def __init__(
        self, 
        num_classes,
        model_type, 
        glove_embeddings, 
        vocab_size, 
        embedding_dim,
        hidden_size, 
        num_layers, 
        dropout, 
        bidirectional
        ):
        
        super(SeqModel, self).__init__()
        self.model_type = model_type

        self.embedding = nn.Embedding.from_pretrained(glove_embeddings)
        self.embedding.weight.requires_grad=False
        # TODO: remember to make remove grad calculation and set training to false
        
        if model_type == "GRU":
            MODEL = nn.GRU
        elif model_type == "LSTM":
            MODEL = nn.LSTM
        self.seqmodel = MODEL(input_size = embedding_dim, hidden_size = hidden_size, batch_first = True, dropout = dropout, num_layers = num_layers , bidirectional = bidirectional)
        
        self.linear= nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        x = self.embedding(x)
        if self.model_type == "GRU":
            _, x = self.seqmodel(x)
        elif self.model_type == "LSTM":
            _, (x, c) = self.seqmodel(x)
        x = self.linear(x)[0]
        return x


def get_glove(config):
    glove_path = config['GLOVE_PATH']
    glove_embeddings = np.load(glove_path)
    glove_embeddings = torch.from_numpy(glove_embeddings)
    return glove_embeddings


def get_model(config):
    return SeqModel(
        config["NUM_CLASSES"],
        config["MODEL_TYPE"],
        get_glove(config),
        config["VOCAB_SIZE"],
        config["EMBEDDING_SIZE"],
        config["HIDDEN_SIZE"],
        config["NUM_LAYERS"],
        config["DROPOUT"],
        config["BIDIRECTIONAL"],
    )