import torch 
import torch.nn as nn  
import torch.nn.functional as F 


class SeqModel(nn.Module):
    def __init__(
        self, 
        num_classes,
        model_type, 
        vocab_size, 
        embedding_dim, 
        hidden_size, 
        num_layers, 
        dropout, 
        bidirectional
        ):
        
        super(SeqModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_directions = 2 if bidirectional == True else 1
        self.num_layers = num_layers
        self.model_type = model_type

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if model_type == "GRU":
            MODEL = nn.GRU
        elif model_type == "LSTM":
            MODEL = nn.LSTM
        self.seqmodel = MODEL(input_size = embedding_dim, hidden_size = hidden_size, batch_first = True, num_layers = num_layers , bidirectional = bidirectional)
        
        self.linear= nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        x = self.embedding(x)
        if self.model_type == "GRU":
            _, x = self.seqmodel(x)
        elif self.model_type == "LSTM":
            _, (x, c) = self.seqmodel(x)
        x = self.linear(x)[0]
        return x


def get_model(config):
    return SeqModel(
        config["NUM_CLASSES"],
        config["MODEL_TYPE"],
        config["VOCAB_SIZE"],
        config["EMBEDDING_SIZE"],
        config["HIDDEN_SIZE"],
        config["NUM_LAYERS"],
        config["DROPOUT"],
        config["BIDIRECTIONAL"],
    )