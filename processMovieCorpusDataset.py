# Malashchuk Vladyslav
# Convert dataset in JSON to question-answer
import json
from collections import defaultdict
import os
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

def print_phase(phase_number, message):
    """Prints a formatted phase message with a decorative border."""
    print(Fore.CYAN + Style.BRIGHT + f"\n{'=' * 10} Phase {phase_number}: {message} {'=' * 10}\n" + Style.RESET_ALL)

def load_config(config_file):
    """Load configuration from a JSON file."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(Fore.GREEN + "Configuration loaded.")
        return config
    except FileNotFoundError:
        print(Fore.RED + "❌ Configuration file not found. Please check the path.")
        raise
    except json.JSONDecodeError:
        print(Fore.RED + "❌ Error decoding JSON from the configuration file.")
        raise

def load_data_from_json(file_path):
    """Load conversation data from a JSON Lines file."""
    lines = {}
    conversations = defaultdict(list)

    print(Fore.YELLOW + f"Loading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_data = json.loads(line.strip())
                line_id = line_data["id"]
                conversation_id = line_data["conversation_id"]
                speaker = line_data["speaker"]
                text = line_data["text"]

                lines[line_id] = {
                    "lineID": line_id,
                    "conversationID": conversation_id,
                    "speaker": speaker,
                    "text": text
                }
                conversations[conversation_id].append(line_id)

        print(Fore.GREEN + f"Loaded {len(lines)} lines and {len(conversations)} conversations.")
        return lines, conversations
    except FileNotFoundError:
        print(Fore.RED + "❌ Data file not found. Please check the path.")
        raise
    except json.JSONDecodeError:
        print(Fore.RED + "❌ Error decoding JSON data.")
        raise

def extract_question_answer_pairs(lines, conversations):
    """Extract question-answer pairs from conversation data."""
    qa_pairs = []

    print_phase(4, "Extracting Question-Answer Pairs")
    for line_ids in conversations.values():
        for i in range(len(line_ids) - 1):  # Iterate through the lines in conversation
            question_line = lines[line_ids[i]]
            answer_line = lines[line_ids[i + 1]]
            qa_pairs.append((question_line["text"], answer_line["text"]))

    print(Fore.GREEN + f"Extracted {len(qa_pairs)} question-answer pairs.")
    return qa_pairs

def save_qa_to_txt(qa_pairs, output_file):
    """Save the extracted question-answer pairs to a text file."""
    print_phase(5, "Saving Question-Answer Pairs")
    print(Fore.YELLOW + f"Saving question-answer pairs to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for question, answer in qa_pairs:
                f.write(f"{question}\t{answer}\n")  # Tab-separated format

        print(Fore.GREEN + "Question-answer pairs saved successfully.")
    except IOError as e:
        print(Fore.RED + f"❌ Error writing to file: {e}")
        raise

def process_dataset(config_file):
    """Main function to process the dataset based on the configuration."""
    try:
        config = load_config(config_file)
        dataset_file_path = os.path.join(config["dataset_folder"], config["dataset_name"], "utterances.jsonl")  # Adjust path if necessary
        lines, conversations = load_data_from_json(dataset_file_path)
        qa_pairs = extract_question_answer_pairs(lines, conversations)
        output_file_path = os.path.join(config["dataset_folder"], config["dataset_name"] + "_qa_output.txt")  # Output path
        save_qa_to_txt(qa_pairs, output_file_path)
        print(Fore.BLUE + f"Question-Answer pairs saved to {output_file_path}.")
    except Exception as e:
        print(Fore.RED + f"❌ An unexpected error occurred: {e}")

# If this module is run directly, the following code will execute
if __name__ == "__main__":
    process_dataset('config.json')  # Replace with your config file path if needed
