import os
import json
import random
from tqdm import tqdm
import argparse
from datasets import load_dataset

def load_tinystories():
    """
    Load the TinyStories dataset using the Hugging Face datasets library.
    """
    print("Loading TinyStories dataset from Hugging Face...")
    dataset = load_dataset("roneneldan/TinyStories")
    return dataset

def preprocess_text(raw_text):
    """
    Clean and preprocess text.
    """
    # Remove excessive newlines
    text = " ".join([line.strip() for line in raw_text.split("\n") if line.strip()])
    
    # Basic cleaning
    text = text.replace("  ", " ")
    
    return text

def extract_samples_from_dataset(dataset, split="train", max_samples=None, min_length=150, max_length=500):
    """
    Extract and preprocess samples from the dataset.
    """
    samples = []
    
    # Get the specified split
    data_split = dataset[split]
    
    print(f"Extracting and preprocessing samples from {split} split...")
    
    # Process each story
    for idx in tqdm(range(len(data_split))):
        if idx >= max_samples and max_samples is not None:
            break
            
        # Get the text sample - typically in a field called 'text' or 'story'
        if 'text' in data_split[idx]:
            story = data_split[idx]['text']
        elif 'story' in data_split[idx]:
            story = data_split[idx]['story']
        else:
            # Try to find the right field by inspecting the first example
            if idx == 0:
                print(f"Available fields: {data_split[0].keys()}")
                # Assume the first field contains the text
                field = list(data_split[0].keys())[0]
                print(f"Using field '{field}' for text")
            story = data_split[idx][field]
            
        # Clean text
        processed_text = preprocess_text(story)
        
        # Filter by length
        if min_length <= len(processed_text) <= max_length:
            samples.append(processed_text)
    
    print(f"Extracted {len(samples)} text samples")
    return samples

def create_dataset_splits(max_samples=50000, min_length=150, max_length=500):
    """
    Create train, validation, and test splits from the dataset.
    """
    # Load the dataset
    dataset = load_tinystories()
    
    # Extract samples from train split
    train_samples = extract_samples_from_dataset(
        dataset, 
        split="train", 
        max_samples=max_samples,
        min_length=min_length,
        max_length=max_length
    )
    
    # Extract samples from validation split if it exists, 
    # otherwise create from train
    if "validation" in dataset:
        val_samples = extract_samples_from_dataset(
            dataset, 
            split="validation", 
            max_samples=int(max_samples * 0.1) if max_samples else None,
            min_length=min_length,
            max_length=max_length
        )
    else:
        # Use 5% of train samples for validation
        val_size = int(len(train_samples) * 0.05)
        val_samples = train_samples[-val_size:]
        train_samples = train_samples[:-val_size]
    
    # Create test set from validation 
    # (or create both from train if no validation exists)
    if len(val_samples) > 100:
        test_size = int(len(val_samples) * 0.5)
        test_samples = val_samples[-test_size:]
        val_samples = val_samples[:-test_size]
    else:
        # Use 5% of remaining train samples for test
        test_size = int(len(train_samples) * 0.05)
        test_samples = train_samples[-test_size:]
        train_samples = train_samples[:-test_size]
    
    return {
        "train": train_samples,
        "validation": val_samples,
        "test": test_samples
    }

def save_splits(splits, output_dir):
    """
    Save the train, validation, and test splits to files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, samples in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}.txt")
        
        print(f"Saving {len(samples)} samples to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(sample + "\n\n")  # Add double newline to separate samples
    
    # Also save a jsonl version for easier loading
    for split_name, samples in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        
        print(f"Saving {len(samples)} samples to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps({"text": sample}) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess the TinyStories dataset")
    parser.add_argument("--processed_dir", default="data/processed", help="Directory to store processed splits")
    parser.add_argument("--max_samples", type=int, default=50000, help="Maximum number of samples to extract")
    parser.add_argument("--min_length", type=int, default=150, help="Minimum text length to include")
    parser.add_argument("--max_length", type=int, default=500, help="Maximum text length to include")
    args = parser.parse_args()

    # Create dataset splits
    splits = create_dataset_splits(
        max_samples=args.max_samples,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    # Save the splits
    save_splits(splits, args.processed_dir)
    
    # Create a small sample for quick testing
    small_splits = {
        "train": splits["train"][:1000],
        "validation": splits["validation"][:100],
        "test": splits["test"][:100]
    }
    small_processed_dir = os.path.join(args.processed_dir, "small")
    save_splits(small_splits, small_processed_dir)
    
    print("Dataset preparation complete!")
    print(f"Train: {len(splits['train'])} samples")
    print(f"Validation: {len(splits['validation'])} samples")
    print(f"Test: {len(splits['test'])} samples")
    print(f"Full dataset in: {args.processed_dir}")
    print(f"Small test dataset in: {small_processed_dir}")

if __name__ == "__main__":
    main()