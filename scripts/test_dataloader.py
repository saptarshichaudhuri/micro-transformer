import torch
import json
import os
import sys

# Add the root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import TransformerDataset
from data import dataloader
from micro_tokenization.tokenizer import MicroTransformerTokenizer
from micro_tokenization.utils import (
    prepare_input_for_batch,
    create_causal_lm_inputs,
    create_masked_lm_inputs,
    test_tokenizer_functionality
)

def test_dataloader(data_loader):
    """
    Test the dataloader by iterating through a few batches.
    """
    print(f"DataLoader has {len(data_loader)} batches")
    
    # Get a few batches
    for i, batch in enumerate(data_loader):
        if i >= 3:  # Just look at first 3 batches
            break
            
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        print(f"Batch {i}:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Label shape: {labels.shape}")
        
        # Verify that inputs and labels are shifted by 1
        match = (input_ids[:, 1:] == labels[:, :-1]).all()
        print(f"  Input and label alignment check: {match}")

# This is where you would call everything
if __name__ == "__main__":
    # 1. Import your tokenizer
    
    # 2. Load your tokenizer
    tokenizer_path = 'micro_tokenization\pretrained'
    tokenizer = MicroTransformerTokenizer.from_pretrained(tokenizer_path)
    
    # 3. Define paths to your data
    data_dir = "data/processed"
    train_path = os.path.join(data_dir, "small", "train.jsonl")  # Using small dataset for testing
    val_path = os.path.join(data_dir, "small", "validation.jsonl")
    
    # 4. Create the dataloaders
    train_loader, val_loader = dataloader.create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        tokenizer=tokenizer,
        batch_size=4,  # Small batch size for testing
        seq_length=64,  # Shorter sequences for testing
        num_workers=0   # Start with 0 workers for testing
    )
    
    # 5. Test the dataloaders
    print("Testing training dataloader:")
    test_dataloader(train_loader)
    
    print("\nTesting validation dataloader:")
    test_dataloader(val_loader)