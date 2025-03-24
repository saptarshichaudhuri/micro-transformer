# dataset.py
# Implementation for the micro-transformer project

import torch
from torch.utils.data import Dataset
import json
import os

class TransformerDataset(Dataset):
    """
    Dataset for training transformer models on text data.
    """
    def __init__(self, data_path, tokenizer, seq_length=128, is_jsonl=True):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file (.jsonl or .txt)
            tokenizer: Tokenizer object with encode method
            seq_length: Maximum sequence length for model input
            is_jsonl: Whether the data is in JSONL format (True) or plain text (False)
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.is_jsonl = is_jsonl
        
        # Load all text samples
        self.samples = []
        
        if is_jsonl:
            # Load JSONL format
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    self.samples.append(item['text'])
        else:
            # Load plain text with double newlines as separators
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
                self.samples = [t.strip() for t in text.split('\n\n') if t.strip()]
        
        # Bug 1001: Making change to the below code to read tokenizer.encode output which is a dict
        # For now only extracting the input_ids from the dict
        # Tokenize all samples and create a single long sequence
        self.tokenized_data = []
        for sample in self.samples:
             # Use the tokenizer and extract input_ids from the returned dict
            tokens_dict = self.tokenizer.encode(sample)
            # Extract input_ids from the dictionary
            input_ids = tokens_dict["input_ids"]
            # Add the input_ids to our sequence
            self.tokenized_data.extend(input_ids)
            
        # Calculate the number of sequences we can create
        self.num_sequences = max(0, len(self.tokenized_data) - self.seq_length)
    
    def __len__(self):
        """Return the number of sequences in the dataset."""
        return self.num_sequences
    
    def __getitem__(self, idx):
        """Get a sequence and its target (next token prediction)."""
        # Get sequence of tokens starting at index idx
        input_sequence = self.tokenized_data[idx:idx + self.seq_length]
        
        # For causal language modeling, target is the same sequence shifted by 1
        target_sequence = self.tokenized_data[idx + 1:idx + self.seq_length + 1]
        
        # Convert to tensors
        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        target_tensor = torch.tensor(target_sequence, dtype=torch.long)
        
        return {
            'input_ids': input_tensor,
            'labels': target_tensor
        }