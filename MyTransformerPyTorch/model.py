import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super(Transformer, self).__init__()

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout) 

        # Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        # Nx Encoders
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Linear output
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src_tokens, tgt_tokens):
        src_seq_length = src_tokens.size(1)
        tgt_seq_length = tgt_tokens.size(1)

        src_mask = (src_tokens == 50258)

        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_length).to(tgt_tokens.device) # decoder mask

        memory_mask = None  # mask between

        src_embeddings = self.token_embedding(src_tokens) 
        src_embeddings = self.positional_encoding(src_embeddings)

        tgt_embeddings = self.token_embedding(tgt_tokens) 
        tgt_embeddings = self.positional_encoding(tgt_embeddings)
        
        # encoder ...
        memory = self.transformer_encoder(src_embeddings.permute(1, 0, 2), src_key_padding_mask=src_mask) #mask=attn_mask)  # [src_seq_length, batch_size, d_model]

        # output of transformer
        output = self.transformer_decoder(
            tgt_embeddings.permute(1, 0, 2), memory, tgt_mask, memory_mask
        )  # [tgt_seq_length, batch_size, d_model]

        output = self.fc_out(output.permute(1, 0, 2))  # [batch_size, tgt_seq_length, vocab_size]
        #output_probs = torch.softmax(output, dim=-1)
        return output

    # look-ahead mask
    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)