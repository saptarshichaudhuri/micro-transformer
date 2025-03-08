import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from micro_tokenization.tokenizer import MicroTransformerTokenizer
from micro_tokenization.utils import test_tokenizer_functionality

def create_training_file(input_files, output_file, max_examples=None):
    """
    Create a single file with all texts for tokenizer training.
    
    Args:
        input_files: List of input files (.jsonl)
        output_file: Path to output file
        max_examples: Maximum number of examples to use for training
    """
    count = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for input_file in input_files:
            with open(input_file, "r", encoding="utf-8") as in_f:
                for line in tqdm(in_f, desc=f"Processing {input_file.name}"):
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        out_f.write(text + "\n")
                        count += 1
                        if max_examples is not None and count >= max_examples:
                            return count
    return count

def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer for MicroTransformer")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the tokenizer")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Vocabulary size")
    parser.add_argument("--min_frequency", type=int, default=2, help="Minimum token frequency")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to use for training")
    args = parser.parse_args()
    
    # Create directories
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find training files - look in the main directory and processed subdirectory
    train_files = list(data_dir.glob("train*.jsonl"))
    
    # If no files found, check processed subdirectory
    if not train_files:
        processed_dir = data_dir / "processed"
        if processed_dir.exists():
            train_files = list(processed_dir.glob("train*.jsonl"))
            
            # Also try other common patterns if needed
            if not train_files:
                train_files = list(processed_dir.glob("*.jsonl"))
    
    if not train_files:
        print(f"No training files (.jsonl) found in {data_dir} or {data_dir}/processed")
        
        # Check for .txt files as fallback
        txt_files = list(data_dir.glob("**/*.txt"))
        if txt_files:
            print(f"Found {len(txt_files)} .txt files instead. Using those.")
            train_files = txt_files
        else:
            print("No suitable training files found.")
            return
    
    print(f"Found {len(train_files)} training files:")
    for file in train_files:
        print(f"  - {file}")
    
    # Create a temporary file with all training texts
    temp_file = data_dir / "tokenizer_train_data.txt"
    
    num_examples = create_training_file(train_files, temp_file, args.max_examples)
    print(f"Created training file with {num_examples} examples")
    
    # Initialize and train tokenizer
    print(f"Training tokenizer with vocab size {args.vocab_size}...")
    tokenizer = MicroTransformerTokenizer()
    tokenizer.train(
        files=[temp_file],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True
    )
    
    # Save the tokenizer
    tokenizer.save(output_dir)
    print(f"Tokenizer saved to {output_dir}")
    
    # Test tokenizer with sample texts
    sample_texts = [
        "Once upon a time in a land far, far away, there lived a tiny dragon.",
        "The quick brown fox jumps over the lazy dog.",
        "I love programming with Python and PyTorch!",
        "This is a test with special characters: @#$%^&*()_+",
        "Let's see how the tokenizer handles this text with punctuation and numbers like 42 and 3.14159."
    ]
    
    # Load the tokenizer from saved files to ensure it works
    loaded_tokenizer = MicroTransformerTokenizer.from_pretrained(output_dir)
    test_tokenizer_functionality(loaded_tokenizer, sample_texts)
    
    # Clean up
    os.remove(temp_file)
    print(f"Temporary file {temp_file} removed")
    
    # Print tokenizer info
    vocab = loaded_tokenizer.get_vocab()
    print(f"Tokenizer vocabulary size: {len(vocab)}")
    print(f"Sample tokens: {list(vocab.items())[:10]}")
    
    # Encode example to show token IDs
    example = "Hello, this is a test sentence for our micro-transformer model."
    encoded = loaded_tokenizer.encode(example)
    print(f"\nExample sentence: {example}")
    print(f"Encoded tokens: {encoded['input_ids']}")
    print(f"Decoded back: {loaded_tokenizer.decode(encoded['input_ids'])}")

if __name__ == "__main__":
    main()