# Malashchuk Vladyslav
# Loads dataset from a server, unzip it, and delete temp zip file

import json
import os
from convokit import Corpus, download
from colorama import Fore, Style, init

# Initialize colorama for cross-platform compatibility
init(autoreset=True)

def load_config(config_file):
    """Load configuration from a JSON file."""
    try:
        with open(config_file, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError:
        print(Fore.RED + "❌ Configuration file not found. Please check the path.")
        raise
    except json.JSONDecodeError:
        print(Fore.RED + "❌ Error decoding JSON from the configuration file.")
        raise

def print_phase(phase_number, message):
    """Prints a formatted phase message with a decorative border."""
    print(Fore.CYAN + Style.BRIGHT + f"\n{'=' * 10} Phase {phase_number}: {message} {'=' * 10}\n" + Style.RESET_ALL)

def download_dataset(config):
    """Download and extract the dataset based on the configuration."""
    dataset_name = config["dataset_name"]
    data_dir = config["dataset_folder"]
    zip_path = os.path.join(data_dir, f"{dataset_name}.zip")

    print_phase(1, "Downloading Dataset")
    try:
        print(f"Downloading and extracting dataset '{dataset_name}' to:", data_dir)
        corpus = Corpus(filename=download(dataset_name, data_dir=data_dir))
        print(Fore.GREEN + "✔ Dataset downloaded and extracted successfully.")

        # Check if the dataset folder exists
        dataset_folder = os.path.join(data_dir, dataset_name)
        if os.path.exists(dataset_folder):
            print(Fore.GREEN + f"✔ Dataset successfully extracted to {dataset_folder}.")
        else:
            print(Fore.YELLOW + "Warning: Dataset folder not found after extraction.")

        return zip_path

    except Exception as e:
        print(Fore.RED + f"❌ An error occurred while downloading the dataset: {e}")
        raise

def cleanup(zip_path):
    """Delete the zip file after extraction."""
    print_phase(2, "Cleaning Up")
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(Fore.GREEN + "✔ Zip file of downloaded dataset deleted successfully.")
    else:
        print(Fore.YELLOW + "Zip file does not exist, skipping deletion.")

def process_dataset(config_file):
    """Main function to load configuration and process dataset."""
    try:
        config = load_config(config_file)
        zip_path = download_dataset(config)
        cleanup(zip_path)

        print(Fore.CYAN + Style.BRIGHT + "\n✔ Dataset downloading completed successfully!\n" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"❌ An unexpected error occurred: {e}")

# If this module is run directly, the following code will execute
if __name__ == "__main__":
    process_dataset("config.json")  # Replace with your config file path if needed
