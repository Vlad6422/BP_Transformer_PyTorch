# Author: Malashchuk Vladyslav
# File: Doctor_AI_dataset.py
# Description: This file contains the implementation of processing the Doctor AI dataset from json to txt format.

import json
from datasets import load_dataset

def process_json_to_txt(input_json, output_file):
    with open(input_json, "r", encoding="utf-8") as infile:
        lines = infile.readlines()


    with open(output_file, "w", encoding="utf-8") as outfile:
        for line in lines:
            try:

                entry = json.loads(line.strip())
                question = entry['Description']
                answer = entry['Doctor']
                
                question = question.lstrip("Q. ").strip()
                

                outfile.write(f"{question}\t{answer}\n")
                

            except json.JSONDecodeError as e:
                print(f"Error on line: {line}. Error: {e}")
    

ds = load_dataset("ruslanmv/ai-medical-chatbot")

ds['train'].to_json('train_data.json')

input_json = "train_data.json"
output_file = "..//datasets/Doctor_AI.txt"

process_json_to_txt(input_json, output_file)
