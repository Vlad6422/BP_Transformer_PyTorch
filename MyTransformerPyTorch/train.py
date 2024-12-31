import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import Transformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import time
from dataset import process_tab_separated_file,ChatDataset
import torch
print(torch.__version__)  # Выводит версию PyTorch
print(torch.cuda.is_available())  # Проверяет доступность CUDA
print(torch.version.cuda)  # Выводит версию CUDA, поддерживаемую текущей версией PyTorch

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Extract model parameters
vocab_size = config["vocab_size"] # Not using right now, Gpt-2 vocab size
d_model = config["d_model"]
nhead = config["nhead"]
num_encoder_layers = config["num_encoder_layers"]
num_decoder_layers = config["num_decoder_layers"]
dim_feedforward = config["dim_feedforward"]
max_seq_length = config["max_seq_length"]
dropout = config["dropout"]

# Extract training parameters
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
num_epochs = config["num_epochs"]
device = torch.device(config["device"])

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

special_tokens = {"bos_token": "<sos>", "pad_token": "<pad>"}
tokenizer.add_special_tokens(special_tokens)

sos_token = tokenizer.bos_token_id  # ID for <sos>
eos_token = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id

print("PAD : ", pad_token_id)
print("SOS : ", sos_token)
print("EOS : ", eos_token)

questions, answers = process_tab_separated_file("movie-corpus_qa_output.txt")

# Print the results for verification
print(f"Total questions: {len(questions)}")
print(f"Total answers: {len(answers)}")

# Define model and move to GPU if available (vocab_size + 2 -- SOS token and PAD)
model = Transformer(vocab_size=tokenizer.vocab_size + 2, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, max_seq_length=max_seq_length,dropout=dropout)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

train_questions, val_questions, train_answers, val_answers = train_test_split(questions, answers, test_size=0.2, random_state=42)

train_dataset = ChatDataset(train_questions, train_answers, tokenizer, sos_token, eos_token, padding_value=pad_token_id, max_length=max_seq_length)
val_dataset = ChatDataset(val_questions, val_answers, tokenizer, sos_token, eos_token, padding_value=pad_token_id, max_length=max_seq_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

print("DATA LOADED!")

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):

        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        print()
        output = model(src, trg[:, :-1])
        
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output_reshape, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    
        output_words = output.argmax(dim=-1)
        output_words = output_words.cpu().numpy()
        trg = trg.cpu().numpy()

        predicted_sentence = tokenizer.decode(output_words[0], skip_special_tokens=False)  
        #predicted_sentence += tokenizer.decode(output_words[1], skip_special_tokens=True)  
        #predicted_sentence += tokenizer.decode(output_words[2], skip_special_tokens=True)  
        #predicted_sentence += tokenizer.decode(output_words[3], skip_special_tokens=True)  
        #predicted_sentence += tokenizer.decode(output_words[4], skip_special_tokens=True)  
        #predicted_sentence += tokenizer.decode(output_words[5], skip_special_tokens=True)  
        #predicted_sentence += tokenizer.decode(output_words[6], skip_special_tokens=True)  
        #predicted_sentence += tokenizer.decode(output_words[7], skip_special_tokens=True)  

        actual_sentence = tokenizer.decode(trg, skip_special_tokens=True)  

        print(f"Step {i}/{len(iterator)} - Loss: {loss.item():.4f}")
        print(f"Predicted: {predicted_sentence}")
        print(f"Actual: {actual_sentence}")
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

best_loss = float('inf')

train_losses, val_losses = [], []
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss = train(model, train_dataloader, optimizer, loss_fn, clip=1.0)
    val_loss = evaluate(model, val_dataloader, loss_fn)
    end_time = time.time()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    print(f'Epoch: {epoch+1} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\tVal Loss: {val_loss:.3f} | Val PPL: {math.exp(val_loss):7.3f}')
    
    # Save model if it's the best so far
    #if val_loss < best_loss:
    print("Model Saved")
    best_loss = val_loss
    torch.save(model.state_dict(), 'saved/model.pt')

   
    plt.figure()
    plt.plot(range(1, epoch+2), train_losses, label='Train Loss')
    plt.plot(range(1, epoch+2), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Epoch {epoch+1} Losses')
    plt.legend()
    plt.savefig(f'result/loss_plot_epoch_{epoch+1}.png')

# Save the losses
with open('result/train_loss.txt', 'w') as f:
    f.write(str(train_losses))

with open('result/val_loss.txt', 'w') as f:
    f.write(str(val_losses))