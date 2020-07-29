import torch 
import torch.nn as nn  
import torch.nn.functional as F 

class SeqModel(nn.Module):
    def __init__(
        self, 
        model_type, 
        vocab_size, 
        embedding_dim, 
        hidden_size, 
        num_layers, 
        dropout, 
        bidirectional
        ):
        
        super(SeqModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if model_type == "RNN":
            MODEL = nn.RNN
        elif model_type == "GRU":
            MODEL = nn.GRU
        elif model_type == "LSTM":
            MODEL = nn.LSTM
        self.seqmodel = MODEL(input_size = embedding_dim, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout, bidirectional = bidirectional)
            
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, prev_state):
        x = self.embedding(x)
        x, state = self.seqmodel(x, prev_state)
        x = self.linear(x)
        return x, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))