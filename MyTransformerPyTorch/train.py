# Author: Malashchuk Vladyslav
# File: train.py
# Description: This file contains train

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
from dataset import process_tab_separated_file, ChatDataset
from metrics import calculate_bleu,calculate_rouge
import pandas as pd

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Extract model parameters
vocab_size = config["vocab_size"]  # Not using right now, GPT-2 vocab size
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

questions, answers = process_tab_separated_file(config["dataset_path"])

# Print the results for verification
print(f"Total questions: {len(questions)}")
print(f"Total answers: {len(answers)}")

# Define model and move to GPU if available
model = Transformer(
    vocab_size=tokenizer.vocab_size + 2,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    max_seq_length=max_seq_length,
    dropout=dropout
)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

train_questions, val_questions, train_answers, val_answers = train_test_split(questions, answers, test_size=0.2, random_state=42)

train_dataset = ChatDataset(train_questions, train_answers, tokenizer, sos_token, eos_token, padding_value=pad_token_id, max_length=max_seq_length)
val_dataset = ChatDataset(val_questions, val_answers, tokenizer, sos_token, eos_token, padding_value=pad_token_id, max_length=max_seq_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

print("DATA LOADED!")

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',        
    factor=0.5,        
    patience=2,       
    verbose=True
)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        trg_input = trg[:, :-1]
        trg_target = trg[:, 1:].contiguous().view(-1)  # Target without <sos>
        output = model(src, trg_input)
        output_reshape = output.contiguous().view(-1, output.shape[-1])

        loss = criterion(output_reshape, trg_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        print(f"Step {i}/{len(iterator)} - Loss: {loss.item():.4f}")
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            trg_input = trg[:, :-1]
            trg_target = trg[:, 1:].contiguous().view(-1)
            output = model(src, trg_input)
            output_reshape = output.contiguous().view(-1, output.shape[-1])

            loss = criterion(output_reshape, trg_target)
            epoch_loss += loss.item()
            print(f"Step {i}/{len(iterator)} - Loss: {loss.item():.4f}")
            
    return epoch_loss / len(iterator)


best_loss = float('inf')
train_losses, val_losses,bleu_scores,rouge_scores = [], [],[],[]

for epoch in range(num_epochs):
    start_time = time.time()

    train_loss = train(model, train_dataloader, optimizer, loss_fn, clip=1.0)
    val_loss = evaluate(model, val_dataloader, loss_fn)
    val_bleu = calculate_bleu(val_dataloader, model, tokenizer, sos_token, eos_token, pad_token_id,device)
    val_rouge = calculate_rouge(val_dataloader, model, tokenizer, sos_token, eos_token, pad_token_id,device)
    
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    end_time = time.time()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    bleu_scores.append(val_bleu)
    rouge_scores.append(val_rouge)

    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
    print(f'Epoch: {epoch+1} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s | Learning Rate: {current_lr:.6f}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\tVal Loss: {val_loss:.3f} | Val PPL: {math.exp(val_loss):7.3f}')
    print(f'\tVal BLEU: {val_bleu:.3f} | Val ROUGE: {val_rouge:.3f}')
    torch.save(model.state_dict(), f'{config["save_model_path"]}_{epoch+1}.pt')

    plt.figure()
    plt.plot(range(1, epoch+2), train_losses, label='Train Loss')
    plt.plot(range(1, epoch+2), val_losses, label='Validation Loss')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(f'Loss over Epochs', fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig(f'result/loss_plot_epoch_{epoch+1}.png')
    plt.close()


    plt.figure()
    plt.bar(range(1, epoch+2), bleu_scores, label='Validation BLEU', color='skyblue')
    plt.xlabel('Epochs')
    plt.ylabel('BLEU Score')
    plt.title(f'Epoch {epoch+1} BLEU Score')
    plt.legend()
    plt.savefig(f'result/bleu_plot_epoch_{epoch+1}.png')
    plt.close()


    plt.figure()
    plt.bar(range(1, epoch+2), rouge_scores, label='Validation ROUGE', color='lightcoral')
    plt.xlabel('Epochs')
    plt.ylabel('ROUGE Score')
    plt.title(f'Epoch {epoch+1} ROUGE Score')
    plt.legend()
    plt.savefig(f'result/rouge_plot_epoch_{epoch+1}.png')
    plt.close()

results = {
    'Epoch': list(range(1, num_epochs+1)),
    'Train Loss': train_losses,
    'Train PPL': [math.exp(x) for x in train_losses],  
    'Val Loss': val_losses,
    'Val PPL': [math.exp(x) for x in val_losses],      
    'Val BLEU': bleu_scores,
    'Val ROUGE': rouge_scores
}

df = pd.DataFrame(results)

df.to_csv('result/training_results.csv', index=False)

print(df)

with open('result/train_loss.txt', 'w') as f:
    f.write(str(train_losses))

with open('result/val_loss.txt', 'w') as f:
    f.write(str(val_losses))

with open('result/bleu_scores.txt', 'w') as f:
    f.write(str(bleu_scores))

with open('result/rouge_scores.txt', 'w') as f:
    f.write(str(rouge_scores))