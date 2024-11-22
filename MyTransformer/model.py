import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=50):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = self.create_positional_encoding(max_len, self.d_model)
        
    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, encoded_words):
        encoded_words = encoded_words.to(device)
        
        embedding = self.embed(encoded_words) * math.sqrt(self.d_model)
        embedding += self.pe[:, :embedding.size(1)]
        embedding = self.dropout(embedding)
        return embedding