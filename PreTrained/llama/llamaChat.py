import torch
from transformers import pipeline


device = 0 if torch.cuda.is_available() else -1

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device=device
)
messages = [
    {"role": "system", "content": "Your name is Llama."},
]

model = pipe.model
eos_token_id = model.config.eos_token_id
pipe.model.config.pad_token_id = eos_token_id
while True:
     
    message = input("You: ")
    messages.append({"role": "user", "content": message})


    outputs = pipe(
        messages,
        max_new_tokens=64,
    )
    print("\n")
    #print(outputs[0]["generated_text"][-1])

    response = outputs[0]["generated_text"][-1]


    #role = response['role']
    content = response['content']
    print(f"You: ", message)


    print(f"Llama: {content}")
