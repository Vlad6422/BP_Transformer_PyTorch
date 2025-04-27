# Author: Malashchuk Vladyslav
# File: chat.py
# Description: This file contains the implementation of chatting with the trained neural network model.

import random
import json
import torch
from model import NeuralNet
from preProcessing import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("config.json", "r") as f:
    config = json.load(f)
# Load intents from JSON file
with open(config["dataset_path"], 'r') as json_data:
    intents = json.load(json_data)

# Load trained model data
FILE = config["save_model_path"]+".pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize and load model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"
print("Let's chat! (type 'quit' to exit)")
# Keep the conversation going until the user types 'quit'
while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    # Tokenize the input sentence
    sentence = tokenize(sentence)
    # Stem each word and create a bag of words
    X = bag_of_words(sentence, all_words)
    # Convert to tensor
    X = X.reshape(1, X.shape[0])
    # Convert to float tensor and move to device
    X = torch.from_numpy(X).to(device)

    # Get the model's prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    # Get the predicted tag and its probability
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.8:  
        
        all_responses = []
        for intent in intents['intents']:
            if tag == intent["tag"]:
                all_responses.extend(intent['responses'])

        if all_responses:
            response = random.choice(all_responses)
            print(f"{bot_name}: {response}")
        else:
            print(f"{bot_name}: I understand the tag '{tag}', but no responses are available.")
    else:
        print(f"{bot_name}: I do not understand...")