# Author: Malashchuk Vladyslav
# File: train.py
# Description: This file contains the implementation of training the neural network model for a chatbot.

import numpy as np
import random
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from preProcessing import bag_of_words, tokenize, stem
from model import NeuralNet

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Load intents
with open('output.json', 'r') as f:
    intents = json.load(f)

# Prepare the dataset
all_words = []
tags = []
xy = []

# Loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters from config
num_epochs = config['num_epochs']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
input_size = len(X_train[0])
hidden_size = config['hidden_size']
output_size = len(tags)
print(input_size, output_size)

# Create a dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Create the dataset
dataset = ChatDataset()

# Split the dataset into training and validation sets
train_size = int(config['train_split'] * len(dataset))  # Percentage for training
val_size = len(dataset) - train_size                     # Remaining for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store loss and accuracy
train_losses = []
val_losses = []
val_accuracies = []

# Early stopping parameters
patience = config['early_stopping']['patience']  # Number of epochs to wait for improvement
best_val_loss = float('inf')                      # Initialize best validation loss
patience_counter = 0                                # Counter for patience

# Train the model with validation
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate the model
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for (words, labels) in val_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            val_loss += criterion(outputs, labels).item()  # Accumulate validation loss

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average validation loss and accuracy
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    # Store losses and accuracy
    train_losses.append(loss.item())
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  # Reset patience counter
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}% (Improvement!)')
    else:
        patience_counter += 1
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # Early stopping
    if patience_counter >= patience:
        print(f'Early stopping triggered after {epoch + 1} epochs.')
        break

# Save the model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

# After plotting training and validation loss

# First figure for loss (larger size)
plt.figure()  # Increased size for better visibility
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.title('Loss over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()
plt.grid()

plt.savefig('loss_plot.png', dpi=300)
plt.show()

# Second figure for accuracy (larger size)
plt.figure()  # Increased size for better visibility
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='orange')
plt.title('Validation Accuracy over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.legend()
plt.grid()

# Save the accuracy plot
plt.tight_layout()
plt.savefig('accuracy_plot.png', dpi=300)
plt.show()
