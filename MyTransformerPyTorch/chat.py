import torch
import json
from transformers import AutoTokenizer
from model import Transformer

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def load_model(config, model_path, tokenizer):
    model = Transformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        max_seq_length=config["max_seq_length"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(model_path, map_location=config["device"]))
    model = model.to(config["device"])
    model.eval()
    return model

def generate_response(model, tokenizer, src_text, config, sos_token, eos_token):
    # Set the model to evaluation mode
    model.eval()

    # Tokenize the input text (source text)
    src_tokens = tokenizer.encode(src_text, add_special_tokens=False)
    print(f"Source tokens: {src_tokens}")  # Debugging output
    src_tokens = torch.tensor(src_tokens).unsqueeze(0).to(config["device"])

    # Initialize the target sequence with the SOS token
    generated_tokens = torch.full((src_tokens.size(0), 1), sos_token, dtype=torch.long).to(config["device"])
    print(sos_token)
    # Autoregressive generation loop
    for _ in range(config.get("max_seq_length", 50)):
        with torch.no_grad():
            # Forward pass through the model
            output = model(src_tokens, generated_tokens)  # [batch_size, tgt_seq_length, vocab_size]
            probs = torch.softmax(output[:, -1, :], dim=-1)
            next_token = torch.argmax(probs, dim=-1)

            # Concatenate the next token to the generated sequence
            generated_tokens = torch.cat((generated_tokens, next_token.unsqueeze(-1)), dim=-1)

            # Debug output
            print(f"Generated tokens: {generated_tokens.squeeze(0).tolist()}")  # Debugging output
            print(f"Next token: {next_token.item()}")  # Debugging output

            # Check if the EOS token is generated, and break if it is
            if next_token.item() == eos_token:
                break

    # Decode the generated tokens back into text
    response_text = tokenizer.decode(generated_tokens.squeeze(0).tolist(), skip_special_tokens=True)
    return response_text
def chat_with_bot(model, tokenizer, config, sos_token, eos_token):
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = generate_response(model, tokenizer, user_input, config, sos_token, eos_token)
        print(f"Bot: {response}")

if __name__ == "__main__":
    # Load configuration
    config_path = "config.json"
    config = load_config(config_path)

    # Setup device
    device = torch.device(config["device"])
    config["device"] = device  # Ensure device is updated

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model_path = "model.pth"

    sos_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id

    model = load_model(config, model_path, tokenizer)

    # Start chatbot
    chat_with_bot(model, tokenizer, config, sos_token+1, eos_token)
