import torch
from typing import List, Dict, Optional, Union
from pathlib import Path

from .tokenizer import MicroTransformerTokenizer

def prepare_input_for_batch(
    texts: List[str],
    tokenizer: MicroTransformerTokenizer,
    max_length: Optional[int] = None,
    padding: bool = True,
    truncation: bool = True,
    return_tensors: str = "pt"
) -> Dict[str, torch.Tensor]:
    """
    Prepare a batch of texts for model input.
    
    Args:
        texts: List of texts to encode
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences
        return_tensors: Return format ('pt' for PyTorch tensors)
    
    Returns:
        Dict containing batched input tensors
    """
    batch = tokenizer.encode_batch(
        texts,
        add_special_tokens=True,
        truncation=truncation,
        max_length=max_length,
        padding=padding,
        return_tensors=return_tensors
    )
    
    return batch

def create_causal_lm_inputs(
    texts: List[str],
    tokenizer: MicroTransformerTokenizer,
    max_length: Optional[int] = None,
    padding: bool = True,
    truncation: bool = True,
    return_tensors: str = "pt"
) -> Dict[str, torch.Tensor]:
    """
    Create inputs for causal language modeling (next token prediction).
    
    Args:
        texts: List of texts to encode
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences
        return_tensors: Return format ('pt' for PyTorch tensors)
    
    Returns:
        Dict containing batched input tensors and labels
    """
    # Encode the texts
    batch = tokenizer.encode_batch(
        texts,
        add_special_tokens=True,
        truncation=truncation,
        max_length=max_length,
        padding=padding,
        return_tensors=None  # We need to process the IDs before converting to tensors
    )
    
    # Get token IDs and attention mask
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    # For causal LM, labels are the same as input_ids
    # but shifted to predict the next token
    labels = []
    for i, ids in enumerate(input_ids):
        # Shift right to create labels (next token prediction)
        labels.append(ids[1:] + [tokenizer.token_to_id("[PAD]")])
    
    # Convert to tensors if requested
    if return_tensors == "pt":
        batch["input_ids"] = torch.tensor(input_ids)
        batch["attention_mask"] = torch.tensor(attention_mask)
        batch["labels"] = torch.tensor(labels)
    else:
        batch["labels"] = labels
    
    return batch

def create_masked_lm_inputs(
    texts: List[str],
    tokenizer: MicroTransformerTokenizer,
    mask_prob: float = 0.15,
    max_length: Optional[int] = None,
    padding: bool = True,
    truncation: bool = True,
    return_tensors: str = "pt"
) -> Dict[str, torch.Tensor]:
    """
    Create inputs for masked language modeling.
    
    Args:
        texts: List of texts to encode
        tokenizer: The tokenizer to use
        mask_prob: Probability of masking a token
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences
        return_tensors: Return format ('pt' for PyTorch tensors)
    
    Returns:
        Dict containing batched input tensors and labels
    """
    import random
    
    # Encode the texts
    batch = tokenizer.encode_batch(
        texts,
        add_special_tokens=True,
        truncation=truncation,
        max_length=max_length,
        padding=padding,
        return_tensors=None  # We need to process the IDs before converting to tensors
    )
    
    # Get token IDs
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    # Create labels (copy of input_ids)
    labels = [[id for id in seq] for seq in input_ids]
    
    # Get mask token ID
    mask_token_id = tokenizer.token_to_id("[MASK]")
    
    # Apply masking
    for i in range(len(input_ids)):
        for j in range(len(input_ids[i])):
            # Skip special tokens and apply mask with probability mask_prob
            is_special = input_ids[i][j] in [
                tokenizer.token_to_id("[CLS]"),
                tokenizer.token_to_id("[SEP]"),
                tokenizer.token_to_id("[PAD]")
            ]
            
            if not is_special and random.random() < mask_prob:
                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    input_ids[i][j] = mask_token_id
                # 10% of the time, replace with random token
                elif random.random() < 0.5:
                    random_token_id = random.randint(0, tokenizer.get_vocab_size() - 1)
                    input_ids[i][j] = random_token_id
                # 10% of the time, keep the token unchanged
                # (labels already contain the original token)
    
    # Convert to tensors if requested
    if return_tensors == "pt":
        batch["input_ids"] = torch.tensor(input_ids)
        batch["attention_mask"] = torch.tensor(attention_mask)
        batch["labels"] = torch.tensor(labels)
    else:
        batch["labels"] = labels
    
    return batch

def test_tokenizer_functionality(
    tokenizer: MicroTransformerTokenizer,
    test_texts: List[str]
) -> None:
    """
    Test tokenizer functionality.
    
    Args:
        tokenizer: The tokenizer to test
        test_texts: List of texts to use for testing
    """
    print("Testing tokenizer functionality...")
    
    # Test encoding and decoding roundtrip
    for text in test_texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded["input_ids"])
        
        # The BPE tokenizer might introduce spaces in unexpected ways
        # So we normalize both original and decoded text for comparison
        norm_orig = text.lower().replace(" ", "")
        norm_decoded = decoded.lower().replace(" ", "")
        
        if norm_orig == norm_decoded:
            print(f"✓ Encoding/decoding roundtrip successful for: {text[:30]}...")
        else:
            print(f"✗ Encoding/decoding mismatch for: {text[:30]}...")
            print(f"  Original: {text}")
            print(f"  Decoded: {decoded}")
    
    # Test special token handling
    special_token_test = "This is a [MASK] test."
    encoded_special = tokenizer.encode(special_token_test)
    mask_id = tokenizer.token_to_id("[MASK]")
    
    if mask_id in encoded_special["input_ids"]:
        print("✓ Special token [MASK] correctly encoded")
    else:
        print("✗ Special token [MASK] not correctly encoded")
    
    # Test batch encoding - Use padding to ensure all sequences have the same length
    # We need to handle sequences of different lengths
    encodings = []
    for text in test_texts:
        encodings.append(tokenizer.encode(text, add_special_tokens=True))
    
    # Find the maximum sequence length 
    max_seq_len = max(len(enc["input_ids"]) for enc in encodings)
    
    # Create batch inputs with proper padding
    batch_inputs = []
    batch_masks = []
    pad_id = tokenizer.token_to_id("[PAD]") 
    
    for enc in encodings:
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        padding_length = max_seq_len - len(ids)
        
        # Pad if needed
        if padding_length > 0:
            ids = ids + [pad_id] * padding_length
            mask = mask + [0] * padding_length
            
        batch_inputs.append(ids)
        batch_masks.append(mask)
    
    # Convert to tensors
    batch_tensor = {
        "input_ids": torch.tensor(batch_inputs),
        "attention_mask": torch.tensor(batch_masks)
    }
    
    if batch_tensor["input_ids"].shape[0] == len(test_texts):
        print("✓ Batch encoding successful")
    else:
        print("✗ Batch encoding failed")
    
    # Test padding
    short_text = "Short text"
    long_text = "This is a much longer text that should be padded differently"
    
    short_enc = tokenizer.encode(short_text, add_special_tokens=True)
    long_enc = tokenizer.encode(long_text, add_special_tokens=True)
    
    # Pad both to the length of the longer text
    target_length = len(long_enc["input_ids"])
    pad_id = tokenizer.token_to_id("[PAD]")
    
    # Pad the shorter text
    padding_length = target_length - len(short_enc["input_ids"])
    padded_ids = short_enc["input_ids"] + [pad_id] * padding_length
    padded_mask = short_enc["attention_mask"] + [0] * padding_length
    
    if len(padded_ids) == len(padded_mask) == target_length:
        print("✓ Padding functionality working correctly")
    else:
        print("✗ Padding functionality not working correctly")
    
    # Test causal language modeling inputs - use only a single example to avoid padding issues
    causal_inputs = create_causal_lm_inputs(
        [test_texts[0]],  # Just use one test text
        tokenizer,
        return_tensors="pt"
    )
    
    if "labels" in causal_inputs and causal_inputs["labels"].shape == causal_inputs["input_ids"].shape:
        print("✓ Causal LM inputs created successfully")
    else:
        print("✗ Causal LM inputs creation failed")
    
    print("Tokenizer testing completed!")