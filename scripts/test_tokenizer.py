import argparse
import torch
from pathlib import Path
from micro_tokenization.tokenizer import MicroTransformerTokenizer
from micro_tokenization.utils import (
    prepare_input_for_batch,
    create_causal_lm_inputs,
    create_masked_lm_inputs,
    test_tokenizer_functionality
)

def test_tokenizer(tokenizer_path, text=None):
    """
    Test tokenizer functionality with a sample text.
    
    Args:
        tokenizer_path: Path to the tokenizer
        text: Sample text to tokenize (optional)
    """
    # Load the tokenizer
    tokenizer = MicroTransformerTokenizer.from_pretrained(tokenizer_path)
    
    # Set default text if not provided
    if text is None:
        text = "This is a test sentence for the micro-transformer tokenizer."
    
    # Basic encoding
    encoding = tokenizer.encode(text, add_special_tokens=True)
    
    print("\n=== BASIC ENCODING ===")
    print(f"Text: {text}")
    print(f"Encoded tokens: {encoding['input_ids']}")
    print(f"Attention mask: {encoding['attention_mask']}")
    
    # Decode back
    decoded = tokenizer.decode(encoding['input_ids'])
    print(f"Decoded: {decoded}")
    
    # Test with truncation and padding
    print("\n=== TRUNCATION & PADDING ===")
    max_length = 20
    truncated = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=True
    )
    print(f"Truncated to {max_length} tokens: {len(truncated['input_ids'])}")
    print(f"Tokens: {truncated['input_ids']}")
    
    # Batch encoding
    print("\n=== BATCH ENCODING ===")
    texts = [
        "The first example sentence.",
        "A second, slightly longer example with more tokens.",
        "Third and final test example."
    ]
    
    batch = tokenizer.encode_batch(
        texts,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt"
    )
    
    print(f"Batch input shape: {batch['input_ids'].shape}")
    print(f"First sequence: {batch['input_ids'][0]}")
    
    # Causal language modeling
    print("\n=== CAUSAL LM INPUTS ===")
    causal_inputs = create_causal_lm_inputs(
        [text],
        tokenizer,
        return_tensors="pt"
    )
    
    print(f"Input shape: {causal_inputs['input_ids'].shape}")
    print(f"Label shape: {causal_inputs['labels'].shape}")
    
    # Masked language modeling
    print("\n=== MASKED LM INPUTS ===")
    masked_inputs = create_masked_lm_inputs(
        [text],
        tokenizer,
        mask_prob=0.2,
        return_tensors="pt"
    )
    
    print(f"Input shape: {masked_inputs['input_ids'].shape}")
    print(f"Label shape: {masked_inputs['labels'].shape}")
    
    # Count masked tokens
    mask_id = tokenizer.token_to_id("[MASK]")
    masked_count = (masked_inputs['input_ids'] == mask_id).sum().item()
    print(f"Number of masked tokens: {masked_count}")
    
    # Comprehensive test
    print("\n=== COMPREHENSIVE TEST ===")
    test_tokenizer_functionality(tokenizer, [text] + texts)
    
    # Vocabulary stats
    vocab = tokenizer.get_vocab()
    print(f"\nVocabulary size: {len(vocab)}")
    
    # Special tokens
    print("\n=== SPECIAL TOKENS ===")
    special_tokens = ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"{token}: {token_id}")

def main():
    parser = argparse.ArgumentParser(description="Test the MicroTransformer tokenizer")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer directory")
    parser.add_argument("--text", type=str, default=None, help="Sample text to tokenize")
    args = parser.parse_args()
    
    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        print(f"Tokenizer not found at {tokenizer_path}")
        return
    
    test_tokenizer(tokenizer_path, args.text)

if __name__ == "__main__":
    main()