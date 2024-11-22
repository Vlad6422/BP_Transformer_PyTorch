import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

def generate_response(model, src_tokens, max_length=50, sos_token=2, eos_token=3, pad_token_id=0):
    model.eval()
    src_tokens = torch.tensor(src_tokens).unsqueeze(0)  # add batch size
    tgt_tokens = torch.tensor([sos_token]).unsqueeze(0)  # Start

    for _ in range(max_length):
        with torch.no_grad():
            output = model(src_tokens, tgt_tokens)  # [1, tgt_seq_length, vocab_size]
            next_token = output[:, -1, :].argmax(dim=-1)  # Text token choise
            tgt_tokens = torch.cat([tgt_tokens, next_token.unsqueeze(0)], dim=1)  # Add token to answer

        if next_token.item() == eos_token:
            break

    return tgt_tokens.squeeze(0).tolist()  # delete batch size

def pad_collate(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_token_id)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_token_id)
    return src_batch, tgt_batch

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, num_classes, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Token Embending and Positional
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))   # Change to sin cos or smth like that (Positional Encoding)
        
        # Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        # Nx Encoders
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Same for Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Linear output
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, src_tokens, tgt_tokens):
        # mask
        src_seq_length = src_tokens.size(1)
        tgt_seq_length = tgt_tokens.size(1)
        src_mask = None  # encode mask
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_length).to(tgt_tokens.device) # decoder mask
        memory_mask = None  # mask between
        
       
        src_embeddings = self.token_embedding(src_tokens) + self.positional_encoding[:, :src_seq_length, :]
        tgt_embeddings = self.token_embedding(tgt_tokens) + self.positional_encoding[:, :tgt_seq_length, :]
        
        # encoder ...
        memory = self.transformer_encoder(src_embeddings.permute(1, 0, 2), src_mask)  # [src_seq_length, batch_size, d_model]
        
       # output of transformer
        output = self.transformer_decoder(
            tgt_embeddings.permute(1, 0, 2), memory, tgt_mask, memory_mask
        )  # [tgt_seq_length, batch_size, d_model]
        
      
        output = self.fc_out(output.permute(1, 0, 2))  # [batch_size, tgt_seq_length, vocab_size]
        return output
    # look-ahead mask
    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    

class ChatDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])


inputs = [
    [2, 4, 11, 12, 13, 5, 3],  # "Hello, how are you?" -> [<sos>, hello, how, are, you, world, <eos>]
    [2, 4, 7, 11, 13, 3],      # "Hello morning, how are you?" -> [<sos>, hello, morning, how, are, you, <eos>]
    [2, 19, 14, 15, 16, 17, 13, 3],  # "And I am fine, thank you" -> [<sos>, and, I, am, fine, thank, you, <eos>]
    [2, 13, 21, 23, 24, 3],    # "What is your name?" -> [<sos>, what, is, your, name, <eos>]
    [2, 25, 26, 27, 28, 3],    # "My name is John" -> [<sos>, my, name, is, John, <eos>]
]

targets = [
    [2, 17, 13, 19, 34, 3],    # "Thank you and thanks" -> [<sos>, thank, you, and, thanks, <eos>]
    [2, 6, 7, 3],              # "Good morning" -> [<sos>, good, morning, <eos>]
    [2, 6, 16, 17, 3],         # "Good fine, thank" -> [<sos>, good, fine, thank, <eos>]
    [2, 25, 26, 27, 28, 3],    # "My name is John" -> [<sos>, my, name, is, John, <eos>]
    [2, 4, 5, 3],              # "Hello world" -> [<sos>, hello, world, <eos>]
]



pad_token_id = 0
batch_size = 8
learning_rate = 1e-4
num_epochs = 30


dataset = ChatDataset(inputs, targets)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)



model = Transformer(vocab_size=10000, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, max_seq_length=128, num_classes=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)


for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for src_tokens, tgt_tokens in dataloader:

        tgt_input = tgt_tokens[:, :-1]
        tgt_output = tgt_tokens[:, 1:]
        #print(tgt_input)
        #print(tgt_output)

        optimizer.zero_grad()

        # Straight
        output = model(src_tokens, tgt_input)  # [batch_size, tgt_seq_length - 1, vocab_size]
        
        
        output = output.reshape(-1, output.size(-1))  # [batch_size * tgt_seq_length, vocab_size]
        tgt_output = tgt_output.reshape(-1)  # [batch_size * tgt_seq_length]
        loss = loss_fn(output, tgt_output)

        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Avg loss: {total_loss / len(dataloader):.4f}")


src_tokens = [2, 13, 21, 23, 24, 3]  # example: "What is your name?"

# Save Model
torch.save(model.state_dict(), "transformer_model.pth")
print("Model Saved!")

response_tokens = generate_response(model, src_tokens, max_length=20, sos_token=2, eos_token=3)

# Vocabulary of all words
vocab = {
    0: "<pad>", 1: "<unk>", 2: "<sos>", 3: "<eos>", 4: "hello", 5: "world", 6: "good", 
    7: "morning", 8: "evening", 9: "night", 10: "day", 11: "how", 12: "are", 
    13: "you", 14: "I", 15: "am", 16: "fine", 17: "thank", 18: "you", 
    19: "and", 20: "what", 21: "about", 22: "is", 23: "your", 24: "name", 
    25: "my", 26: "name", 27: "is", 28: "John", 29: "yes", 30: "no", 
    31: "maybe", 32: "okay", 33: "please", 34: "thanks", 35: "sorry", 
    36: "excuse", 37: "me", 38: "not", 39: "sure", 40: "can", 41: "you", 
    42: "help", 43: "me", 44: "with", 45: "this", 46: "problem", 47: "love", 
    48: "hate", 49: "like", 50: "dislike", 51: "friend", 52: "family", 
    53: "happy", 54: "sad", 55: "angry", 56: "hungry", 57: "thirsty", 
    58: "tired", 59: "bored", 60: "fun", 61: "interesting", 62: "boring", 
    63: "beautiful", 64: "ugly", 65: "nice", 66: "bad", 67: "awesome", 
    68: "amazing", 69: "wonderful", 70: "horrible", 71: "terrible", 
    72: "happy", 73: "sad", 74: "today", 75: "yesterday", 76: "tomorrow", 
    77: "week", 78: "month", 79: "year", 80: "life", 81: "time", 82: "moment", 
    83: "hour", 84: "minute", 85: "second", 86: "before", 87: "after", 
    88: "now", 89: "later", 90: "always", 91: "never", 92: "sometimes", 
    93: "often", 94: "rarely", 95: "usually", 96: "important", 
    97: "necessary", 98: "possible", 99: "impossible", 100: "yes", 
    101: "no", 102: "sure", 103: "maybe", 104: "try", 105: "do", 106: "go", 
    107: "come", 108: "stay", 109: "run", 110: "walk", 111: "sit", 
    112: "stand", 113: "look", 114: "see", 115: "hear", 116: "listen", 
    117: "speak", 118: "say", 119: "talk", 120: "write", 121: "read", 
    122: "learn", 123: "study", 124: "work", 125: "play", 126: "eat", 
    127: "drink", 128: "sleep", 129: "wake", 130: "up", 131: "down", 
    132: "left", 133: "right", 134: "fast", 135: "slow", 136: "hot", 
    137: "cold", 138: "big", 139: "small", 140: "long", 141: "short", 
    142: "old", 143: "young", 144: "new", 145: "same", 146: "different", 
    147: "good", 148: "better", 149: "best", 150: "bad", 151: "worse", 
    152: "worst", 153: "important", 154: "unimportant", 155: "necessary", 
    156: "unnecessary", 157: "possible", 158: "impossible", 159: "easy", 
    160: "hard", 161: "difficult", 162: "simple", 163: "interesting", 
    164: "boring", 165: "beautiful", 166: "ugly", 167: "clean", 168: "dirty", 
    169: "happy", 170: "sad", 171: "angry", 172: "bored", 173: "excited", 
    174: "tired", 175: "sleepy", 176: "hungry", 177: "thirsty", 
    178: "hot", 179: "cold", 180: "ready", 181: "busy", 182: "free", 
    183: "open", 184: "closed", 185: "on", 186: "off", 187: "yes", 
    188: "no", 189: "ok", 190: "please", 191: "sorry", 192: "thank", 
    193: "you", 194: "welcome", 195: "bye", 196: "hello", 197: "hi", 
    198: "goodbye", 199: "see", 200: "later"
}

response = " ".join([vocab[token] for token in response_tokens if token not in {2, 3, 0}])

print("Bot:", response)