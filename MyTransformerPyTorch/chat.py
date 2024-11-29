import torch
from transformers import AutoTokenizer
from model import Transformer

def load_model(model_path, tokenizer, device):
    
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=2048,
        max_seq_length=512,
        num_classes=0
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def generate_response(model, tokenizer, src_text, max_length=50, sos_token=2, eos_token=3, device='cpu'):
    model.eval()
    src_tokens = tokenizer.encode(src_text, add_special_tokens=False)
    src_tokens = torch.tensor(src_tokens).unsqueeze(0).to(device)
    tgt_tokens = torch.tensor([sos_token]).unsqueeze(0).to(device)

    for _ in range(max_length):
        with torch.no_grad():
            output = model(src_tokens, tgt_tokens)
            next_token = output[:, -1, :].argmax(dim=-1)
            tgt_tokens = torch.cat([tgt_tokens, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == eos_token:
            break

    return tokenizer.decode(tgt_tokens.squeeze(0).tolist(), skip_special_tokens=True)

def chat_with_bot(model, tokenizer, sos_token, eos_token, device):
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = generate_response(model, tokenizer, user_input, max_length=50, sos_token=sos_token, eos_token=eos_token, device=device)
        print(f"Bot: {response}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model_path = "transformer_model.pth"


    sos_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id

    model = load_model(model_path, tokenizer, device)

    chat_with_bot(model, tokenizer, sos_token, eos_token, device)
