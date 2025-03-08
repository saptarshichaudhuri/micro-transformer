"""
Script to download and prepare the TinyStories dataset for training.
"""
import os
import sys
import argparse

# Add the root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.preprocessing import create_dataset_splits, save_splits

def main():
    parser = argparse.ArgumentParser(description="Prepare data for micro-transformer training")
    parser.add_argument("--processed_dir", default="data/processed", help="Directory for processed data")
    parser.add_argument("--max_samples", type=int, default=50000, 
                        help="Maximum number of samples to extract (50k is ~25MB of text)")
    parser.add_argument("--min_length", type=int, default=150, help="Minimum text length to include")
    parser.add_argument("--max_length", type=int, default=500, help="Maximum text length to include")
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.processed_dir, exist_ok=True)
    
    # Create dataset splits
    splits = create_dataset_splits(
        max_samples=args.max_samples,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    # Save splits
    save_splits(splits, args.processed_dir)
    
    # Create a small sample for quick testing
    small_splits = {
        "train": splits["train"][:1000],
        "validation": splits["validation"][:100],
        "test": splits["test"][:100]
    }
    small_processed_dir = os.path.join(args.processed_dir, "small")
    save_splits(small_splits, small_processed_dir)
    
    print("Data preparation complete!")
    print(f"Full dataset in: {args.processed_dir}")
    print(f"Small test dataset in: {small_processed_dir}")

if __name__ == "__main__":
    main()