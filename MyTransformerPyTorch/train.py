import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from torch.cuda.amp import GradScaler, autocast
from model import Transformer

def pad_collate(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_token_id)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_token_id)
    return src_batch, tgt_batch


class ChatDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, sos_token, eos_token):
        self.tokenizer = tokenizer
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.inputs = [self.tokenizer.encode(text, add_special_tokens=False,max_length=128) for text in inputs]
        self.targets = [self.tokenizer.encode(text, add_special_tokens=False,max_length=128) for text in targets]

        # Add SOS and EOS to each target sequence
        self.targets = [[self.sos_token] + target + [self.eos_token] for target in self.targets]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Special tokens for SOS and EOS
sos_token = tokenizer.bos_token_id
eos_token = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id

questions = []  # List to store the questions
answers = []    # List to store the answers
# Open the file and read line by line
# Open the file and read line by line
with open('movie-corpus_qa_output.txt', 'r') as file:
    # Get the total number of lines
    total_lines = sum(1 for _ in file)

    # Reopen the file to start reading
    file.seek(0)
    
    # Read only 10% of the file
    lines_to_read = total_lines // 95
    
    for i, line in enumerate(file):
        # Stop reading if we've reached 10% of the file
        if i >= lines_to_read:
            break
        
        # Strip any leading/trailing whitespace (including newlines)
        line = line.strip()
        
        # Skip lines that don't contain a tab character
        if "\t" not in line:
            continue
        
        # Split the line by the tab character
        question, answer = line.split('\t')
        
        # Append the question and answer to the respective lists
        questions.append(question)
        answers.append(answer)

# Now `questions` and `answers` contain all the question-answer pairs
#print("Questions:", questions)
#print("Answers:", answers)
# Prepare dataset and dataloader
batch_size = 4
learning_rate = 1e-4
num_epochs = 100

pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0  # Default to 0 if not available

dataset = ChatDataset(questions, answers, tokenizer, sos_token, eos_token)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate,drop_last=True,pin_memory=True)
print("DATA LOADED!")
# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and move to GPU if available
model = Transformer(vocab_size=tokenizer.vocab_size, d_model=128, nhead=8, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=2048, max_seq_length=512, num_classes=0)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
print("TRAIN STARTED!")
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    a = 0
    for src_tokens, tgt_tokens in dataloader:
        print(a ,"/", len(dataloader))
        a = a + 1
        # Move data to the same device as the model
        src_tokens = src_tokens.to(device)
        tgt_tokens = tgt_tokens.to(device)

        tgt_input = tgt_tokens[:, :-1]  # Remove EOS token for the input to the decoder
        tgt_output = tgt_tokens[:, 1:]  # Remove SOS token for the target output

        optimizer.zero_grad()

        # Forward pass
        output = model(src_tokens, tgt_input)  # [batch_size, tgt_seq_length - 1, vocab_size]

        output = output.reshape(-1, output.size(-1))  # [batch_size * tgt_seq_length, vocab_size]
        tgt_output = tgt_output.reshape(-1)  # [batch_size * tgt_seq_length]
        loss = loss_fn(output, tgt_output)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Avg loss: {total_loss / len(dataloader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "transformer_model.pth")
print("Model Saved!")