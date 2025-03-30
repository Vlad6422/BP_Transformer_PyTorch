# Author: Malashchuk Vladyslav
# File: ChitChatToJSON.py
# Description: This file contains the implementation of converting the ChitChat dataset from .qna format to .json format.

import json

def txt_to_json(input_file, output_file):
    intents = []
    
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    current_intent = {}
    collecting_responses = False
    
    for line in lines:
        line = line.strip()
        

        if line.startswith("**Filters**") or "- editorial = chitchat" in line.lower():
            continue
        
        if line.startswith("# ?"):
            
            if current_intent:
                intents.append(current_intent)
            
            
            current_intent = {
                "tag": line[3:].strip().replace(" ", "_").lower(),
                "patterns": [],
                "responses": []
            }
            collecting_responses = False
        elif line.startswith("-"):
           
            if "patterns" in current_intent:
                current_intent["patterns"].append(line[1:].strip())
        elif line.startswith("```markdown"):
            
            collecting_responses = True
        elif collecting_responses:
            
            if line.startswith("```") or line.startswith(">"):
                
                collecting_responses = False
            elif line and not line.startswith(">"):
                current_intent["responses"].append(line.strip())

    
    if current_intent:
        intents.append(current_intent)


    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump({"intents": intents}, json_file, indent=4, ensure_ascii=False)


input_file = "./datasets/qna_chitchat_professional.qna"
output_file = "./datasets/output.json"
txt_to_json(input_file, output_file)
print(f"JSON saved to {output_file}")
