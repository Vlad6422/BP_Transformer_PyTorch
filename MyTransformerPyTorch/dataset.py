import torch
from torch.utils.data import Dataset
class ChatDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, sos_token, eos_token, max_length=10, padding_value=0):
        """
        Custom dataset for chat-based tokenized inputs and targets.

        Args:
            inputs (list of str): List of input strings.
            targets (list of str): List of target strings.
            tokenizer (Tokenizer): Tokenizer object for encoding text.
            sos_token (int): Start of sequence token ID.
            eos_token (int): End of sequence token ID.
            max_length (int): Maximum sequence length for inputs and targets.
            padding_value (int): Padding value for sequences.
        """
        self.tokenizer = tokenizer
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_length = max_length
        self.padding_value = padding_value

        # Tokenize inputs and targets
        self.inputs = [self._prepare_sequence(text) for text in inputs]
        self.targets = [self._prepare_sequence(text, is_target=True) for text in targets]

    def _prepare_sequence(self, text, is_target=False):
        """
        Tokenizes and processes a text sequence.

        Args:
            text (str): Input text to tokenize.
            is_target (bool): Whether the sequence is a target (adds SOS/EOS tokens).

        Returns:
            list: Tokenized and processed sequence.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=self.max_length - 2)
        if is_target:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        return tokens

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieves a padded input-target pair.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: Tuple of padded input and target tensors.
        """
        input_tensor = torch.tensor(self._pad_sequence(self.inputs[idx]), dtype=torch.long)
        target_tensor = torch.tensor(self._pad_sequence(self.targets[idx]), dtype=torch.long)
        return input_tensor, target_tensor

    def _pad_sequence(self, sequence):
        """
        Pads a sequence to the maximum length.

        Args:
            sequence (list): Sequence to pad.

        Returns:
            list: Padded sequence.
        """
        return sequence + [self.padding_value] * (self.max_length - len(sequence))
    
def process_tab_separated_file(file_path):
    """
    Processes a tab-separated file to extract questions and answers.

    Args:
        file_path (str): Path to the input file.

    Returns:
        tuple: Two lists, one containing questions and the other containing answers.
    """
    questions = []
    answers = []

    # Open the file and process it
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip leading/trailing whitespace (including newlines)
            line = line.strip()

            # Skip lines without a tab character
            if "\t" not in line:
                continue

            # Split the line into question and answer parts
            try:
                question, answer = line.split('\t', 1)  # Allow only one split
            except ValueError:
                continue  # Skip lines that don't split correctly

            # Append the question and answer to the respective lists
            questions.append(question)
            answers.append(answer)

    return questions, answers

