import torch
from transformers import AutoTokenizer
from model import Transformer
import json
import torch.nn.functional as F 
import argparse

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def load_model(config, model_path, tokenizer):
    model = Transformer(
        vocab_size=config["vocab_size"] + 2,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        max_seq_length=config["max_seq_length"],
        dropout=config["dropout"]
    )
    model.load_state_dict(torch.load(model_path, map_location=config["device"]))
    model = model.to(config["device"])
    model.eval()
    return model

def generate_response(model, tokenizer, src_text, config, sos_token, eos_token, temperature=1.4, top_k=3):
    """
    Generation + Temperature + Top_K
    """

    src_tokens = tokenizer.encode(src_text, add_special_tokens=False)

    max_seq_length = config["max_seq_length"]
    pad_token = 50258

    src_tokens = tokenizer.encode(src_text, add_special_tokens=True)
    src_tokens = torch.tensor(src_tokens).unsqueeze(0).to(config["device"])

    if src_tokens.size(1) < max_seq_length:
        src_tokens = torch.nn.functional.pad(
            src_tokens, 
            (0, max_seq_length - src_tokens.size(1)),
            value=pad_token
        )
    generated_tokens = torch.full((src_tokens.size(0), 1), sos_token, dtype=torch.long).to(config["device"])

    for _ in range(config["max_seq_length"]):
        with torch.no_grad():
            output = model(src_tokens, generated_tokens)
            logits = output[:, -1, :] 

            logits = logits / temperature

            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
                probabilities = F.softmax(top_k_values, dim=-1)
                next_token = top_k_indices.gather(
                    dim=1,
                    index=torch.multinomial(probabilities, num_samples=1)
                ).squeeze(-1)

            else:
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

            generated_tokens = torch.cat((generated_tokens, next_token.unsqueeze(-1)), dim=-1)

            if next_token.item() == eos_token:
                break

    response_text = tokenizer.decode(generated_tokens.squeeze(0).tolist(), skip_special_tokens=True)
    return response_text

def chat_with_bot(model, tokenizer, config, sos_token, eos_token, temperature, top_k):
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = generate_response(model, tokenizer, user_input, config, sos_token, eos_token, temperature, top_k)
        print(f"Bot: {response}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Chat with the Transformer-based chatbot.")
    parser.add_argument('--temperature', type=float, default=1.4, help='Temperature for generation (higher is more random)')
    parser.add_argument('--top_k', type=int, default=3, help='Top-K for sampling (0 means no top-K sampling)')
    parser.add_argument('--config', type=str, default="config.json", help='Path to the config file')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Load the config file
    config = load_config(args.config)
    device = torch.device(config["device"])
    config["device"] = device

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    special_tokens = {"bos_token": "<sos>", "pad_token": "<pad>"}
    tokenizer.add_special_tokens(special_tokens)
    sos_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    # Generate the model path based on the config file (same as before)
    model_path = f'{config["save_model_path"]}_{config["num_epochs"]}.pt'

    # Load the model
    model = load_model(config, model_path, tokenizer)

    # Start the chatbot
    chat_with_bot(model, tokenizer, config, sos_token, eos_token, args.temperature, args.top_k)
