from transformers import T5Tokenizer, T5ForConditionalGeneration

# Загрузка токенизатора и модели
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")

print("Chatbot is ready! Type 'exit' to quit the chat.\n")

dialogue_history = ""

while True:
    input_text = input("You: ")
    
    if input_text.lower() in {"exit", "quit"}:
        print("Chatbot: Goodbye!")
        break
    
    try:
        # add context
        dialogue_history += f"User: {input_text}\nChatbot:"
        
        # generate
        input_ids = tokenizer(dialogue_history, return_tensors="pt", truncation=True, max_length=512).input_ids.to("cuda")
        outputs = model.generate(input_ids, max_length=100, temperature=0.7, top_p=0.9, top_k=50, num_return_sequences=1)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # history/memory
        dialogue_history += f" {response}\n"
        
        print(f"T5: {response}")
    
    except Exception as e:
        print(f"Chatbot: Sorry, I encountered an error: {e}")
