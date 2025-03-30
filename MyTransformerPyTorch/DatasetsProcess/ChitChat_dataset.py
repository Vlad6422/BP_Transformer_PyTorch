# Author: Malashchuk Vladyslav
# File: ChitChat_dataset.py
# Description: This file contains the implementation of processing the ChitChat dataset from qna to txt format.


import re

def convert_qna_to_tsv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the content into sections by the QnA source tag
    sections = re.split(r"> !# @qna\.pair\.source = .+?\n", content)

    qna_pairs = []

    for section in sections:
        
        # Extract answer block in markdown
        answer_match = re.search(r"```markdown\n(.*?)\n```", section, re.DOTALL)
        if not answer_match:
            continue
        answer = answer_match.group(1).strip()

        # Extract all questions in the section
        questions = re.findall(r"-\s*(.+?)\n", section)

        # Pair each question with the answer
        for question in questions:
            if 'editorial = chitchat' in question:
                print(f"Skipping question: {question} (editorial = chitchat)")
                continue

            qna_pairs.append((question, answer))

    if not qna_pairs:
        raise ValueError("No QnA pairs found in the file.")

    # Write QnA pairs to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for question, answer in qna_pairs:
            f.write(f"{question}\t{answer}\n")

# Specify input and output file paths
input_file = 'qna_chitchat_professional.qna'
output_file = '../datasets/ChitChat.txt'

# Convert QnA pairs
convert_qna_to_tsv(input_file, output_file)
print(f"QnA pairs have been written to {output_file}")