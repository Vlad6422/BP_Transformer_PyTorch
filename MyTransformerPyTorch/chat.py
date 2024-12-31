import torch
from transformers import AutoTokenizer
from model import Transformer
import json


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


def generate_response(model, tokenizer, src_text, config, sos_token, eos_token):

    src_tokens = tokenizer.encode(src_text, add_special_tokens=False)
    src_tokens = torch.tensor(src_tokens).unsqueeze(0).to(config["device"])

    generated_tokens = torch.full((src_tokens.size(0), 1), sos_token, dtype=torch.long).to(config["device"])


    for _ in range(config["max_seq_length"]):
        with torch.no_grad():

            output = model(src_tokens, generated_tokens)
            
            probs = torch.softmax(output[:, -1, :], dim=-1)
            print(probs)
            next_token = torch.argmax(probs, dim=-1)
            print(next_token)

            generated_tokens = torch.cat((generated_tokens, next_token.unsqueeze(-1)), dim=-1)
            output_words = output.argmax(dim=-1) 
            output_words = output_words.cpu().numpy()

            predicted_sentence = tokenizer.decode(output_words[0], skip_special_tokens=True)

            print(f"Predicted: {predicted_sentence}")


            if next_token.item() == eos_token:
                break


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

    config_path = "config.json"
    config = load_config(config_path)

    device = torch.device(config["device"])
    config["device"] = device


    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model_path = "saved/model.pt"
    special_tokens = {"bos_token": "<sos>", "pad_token": "<pad>"}
    tokenizer.add_special_tokens(special_tokens)
    sos_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    sos_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id


    model = load_model(config, model_path, tokenizer)

    chat_with_bot(model, tokenizer, config, sos_token, eos_token)
