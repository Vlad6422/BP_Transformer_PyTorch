from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_response(input_text, max_length=10, temperature=0.01, top_k=50):


    # Tokin
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # Answer
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length, 
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            no_repeat_ngram_size=2,
            num_return_sequences=1
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


model_name = "gpt2" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

print("Ready")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Gpt-2: Bye!")
        break
    
    # Asnwer
    response = generate_response(user_input)
    print(f"Gpt-2: {response}")
