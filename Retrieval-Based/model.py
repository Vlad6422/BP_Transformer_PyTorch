# Author: Malashchuk Vladyslav
# File: model.py
# Description: This file contains the implementation of the neural network model for the chatbot.

import torch.nn as nn

# Define the neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Define the layers of the neural network
        self.l1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.1)  
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.1)  
        self.l3 = nn.Linear(hidden_size, num_classes)
        # Define the activation function
        self.relu = nn.ReLU()
    # Define the forward pass
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout1(out)  
        
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout2(out)  
        
        out = self.l3(out)
        return out
