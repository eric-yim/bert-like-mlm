import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class PositionalEncoding(nn.Module):

    def __init__(self,d_model, dropout=0.1, pe_max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(pe_max_len, d_model)
        position = torch.arange(0, pe_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
class TransformerModel(nn.Module):
    """
    transformer with positional encoding
    see https://pytorch.org/tutorials/beginner/transformer_tutorial.html for positional encoding
    see https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html for transformer input shapes
    """
    def __init__(self,main_params,pe_max_len=5000,vocab_size=10):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(**main_params)
        d_model = main_params['d_model']
        dropout = main_params['dropout']
        self.pos_encoder = PositionalEncoding(d_model, dropout,pe_max_len)
        
        self.embedding = nn.Embedding(vocab_size,d_model)
        
        self.d_model = d_model

        self.fc = nn.Linear(d_model,vocab_size)
    def generate_square_subsequent_mask(self, sz):
        return self.transformer.generate_square_subsequent_mask(sz)


    def forward(self, src,**kwargs):

        src = self.embedding(src)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        output = self.transformer(src, src,**kwargs)
        output = self.fc(output) # Do not use softmax if using nn.CrossEntropyLoss()
        return output
        
        
        