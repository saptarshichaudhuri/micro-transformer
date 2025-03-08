# Micro-Transformer Tokenization Phase: Technical Documentation

## 1. Understanding BPE Tokenization

### What is Tokenization?
Tokenization is the process of converting text into smaller units (tokens) that can be processed by a language model. These tokens can be words, subwords, characters, or even bytes. Tokenization is a critical preprocessing step for any NLP model as it defines the vocabulary the model will work with.

### Byte-Pair Encoding (BPE) Tokenization
BPE is a subword tokenization algorithm that strikes a balance between character-level and word-level tokenization. It was originally developed as a data compression technique but was adapted for NLP by Sennrich et al. (2016) for Neural Machine Translation.

#### How BPE Works:
1. **Initialization**: Start with a vocabulary of individual characters/bytes
2. **Frequency Counting**: Count the frequency of adjacent pairs of tokens in the corpus
3. **Merging**: Iteratively merge the most frequent pairs to create new tokens
4. **Stopping Criterion**: Stop when a desired vocabulary size is reached or no more merges are possible

#### ByteLevelBPE:
The ByteLevelBPE variant operates on bytes rather than characters, which makes it encoding-agnostic. It handles any input regardless of the language or presence of special characters. This is the tokenization approach used by models like GPT-2, RoBERTa, and others.

#### Benefits of BPE:
- Handles out-of-vocabulary words by breaking them into known subword units
- Balances vocabulary size with token length
- Works well for morphologically rich languages
- Language-agnostic (especially ByteLevelBPE)

## 2. Project Structure Overview

Our tokenization implementation consists of several components:

```
micro_tokenization/
├── __init__.py          # Package initialization and exports
├── tokenizer.py         # Core tokenizer implementation
└── utils.py             # Helper functions for the tokenizer

scripts/
├── train_tokenizer.py   # Script to train the tokenizer on data
└── test_tokenizer.py    # Script to test tokenizer functionality
```

## 3. Core Components Explained

### 3.1 MicroTransformerTokenizer Class (`tokenizer.py`)

The `MicroTransformerTokenizer` class is a wrapper around Hugging Face's `ByteLevelBPETokenizer` that adds functionality specific to our transformer model needs.

#### Key Features:

1. **Initialization and Loading**:
   - Initialize a new tokenizer or load from pre-trained files
   - The constructor accepts paths to vocabulary and merges files

2. **Special Tokens**:
   - Adds and handles special tokens: [CLS], [SEP], [MASK], [PAD]
   - [CLS] marks the beginning of a sequence
   - [SEP] marks the end of a sequence
   - [MASK] is used for masked language modeling
   - [PAD] is used for padding sequences to a uniform length

3. **Training**:
   - Trains on text files to learn the vocabulary
   - Configurable vocabulary size (default 5000 tokens)
   - Configurable minimum frequency threshold

4. **Encoding**:
   - Converts text to token IDs
   - Supports single text and batch encoding
   - Handles truncation to maximum length
   - Handles padding to make sequences uniform length
   - Optionally returns PyTorch tensors

5. **Decoding**:
   - Converts token IDs back to text
   - Supports both single sequence and batch decoding
   - Option to skip special tokens in the output

6. **Saving and Loading**:
   - Saves tokenizer files (vocabulary and merges)
   - Loads from saved files
   - Includes a configuration file with metadata

7. **Utility Methods**:
   - Get vocabulary size
   - Get the full vocabulary mapping
   - Convert between tokens and their IDs

#### Implementation Details:

```python
def encode(self, text, add_special_tokens=True, truncation=False, max_length=None, padding=False, return_tensors=None):
    # First encode the text using the underlying tokenizer
    encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    # Extract values as mutable lists
    input_ids = list(encoding.ids)
    token_type_ids = list(encoding.type_ids)
    attention_mask = list(encoding.attention_mask)
    
    # Handle truncation if needed
    if truncation and max_length is not None and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        token_type_ids = token_type_ids[:max_length]
        attention_mask = attention_mask[:max_length]
    
    # Handle padding if needed
    if padding and max_length is not None:
        pad_id = self.tokenizer.token_to_id("[PAD]")
        pad_len = max_length - len(input_ids)
        
        if pad_len > 0:
            input_ids.extend([pad_id] * pad_len)
            token_type_ids.extend([0] * pad_len)
            attention_mask.extend([0] * pad_len)
    
    # Create result dictionary
    result = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask
    }
    
    # Convert to tensors if requested
    if return_tensors == "pt":
        import torch
        for key in result:
            result[key] = torch.tensor([result[key]])
    
    return result
```

The `encode_batch` method follows a similar pattern but handles a list of texts instead of a single text.

### 3.2 Utility Functions (`utils.py`)

The utilities module provides helper functions for common tokenization tasks:

1. **prepare_input_for_batch**:
   - Prepares a batch of texts for model input
   - Handles tokenization, padding, and tensor conversion
   - Returns a dictionary ready for model consumption

2. **create_causal_lm_inputs**:
   - Creates inputs specifically for causal language modeling
   - Handles the label shifting for next token prediction
   - Important for autoregressive training

3. **create_masked_lm_inputs**:
   - Creates inputs for masked language modeling (BERT-style)
   - Randomly masks tokens with a configurable probability
   - Creates corresponding labels for the masked tokens

4. **test_tokenizer_functionality**:
   - Comprehensive testing of tokenizer features
   - Verifies encoding/decoding roundtrip
   - Tests special token handling
   - Tests batch encoding and padding

#### Implementation Highlights:

The masked language modeling function uses a strategy similar to BERT:
- 80% of the time, replace the token with [MASK]
- 10% of the time, replace with a random token
- 10% of the time, keep the original token
- Original tokens are preserved in the labels for the model to predict

```python
def create_masked_lm_inputs(texts, tokenizer, mask_prob=0.15, ...):
    # Encode the texts
    batch = tokenizer.encode_batch(texts, ...)
    
    # Get token IDs and create labels
    input_ids = batch["input_ids"]
    labels = [[id for id in seq] for seq in input_ids]
    
    # Get mask token ID
    mask_token_id = tokenizer.token_to_id("[MASK]")
    
    # Apply masking
    for i in range(len(input_ids)):
        for j in range(len(input_ids[i])):
            # Skip special tokens
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
    
    # Rest of the function handles tensor conversion etc.
```

### 3.3 Training Script (`scripts/train_tokenizer.py`)

The training script handles:

1. **Data Collection**:
   - Finds training files in the specified directory
   - Supports jsonl format with "text" fields
   - Creates a temporary concatenated file for training

2. **Tokenizer Training**:
   - Creates a new tokenizer instance
   - Configures vocabulary size and minimum frequency
   - Trains on the collected data

3. **Validation**:
   - Tests the trained tokenizer on sample texts
   - Verifies encoding/decoding works as expected

4. **Saving**:
   - Saves the tokenizer files to the specified directory
   - Includes vocabulary, merges, and configuration

#### Key Function:

```python
def create_training_file(input_files, output_file, max_examples=None):
    """
    Create a single file with all texts for tokenizer training.
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
```

### 3.4 Testing Script (`scripts/test_tokenizer.py`)

The testing script provides:

1. **Comprehensive Testing**:
   - Tests basic encoding/decoding
   - Tests truncation and padding
   - Tests batch encoding
   - Tests causal and masked language modeling inputs

2. **Detailed Output**:
   - Shows examples of encoded tokens
   - Shows attention masks
   - Shows decoded results

3. **Validation**:
   - Verifies special tokens are handled correctly
   - Checks if padding works correctly
   - Tests tensor conversion

## 4. Integration with Transformer Models

The tokenizer we've implemented is designed to work seamlessly with transformer models like the one we're building:

1. **Model Input Format**:
   - The tokenizer produces inputs in the format expected by transformers
   - "input_ids": The token IDs
   - "attention_mask": Mask indicating which tokens are real vs. padding
   - "token_type_ids": Useful for tasks with multiple segments

2. **Training Scenarios**:
   - For autoregressive (causal) language modeling, our `create_causal_lm_inputs` function shifts the labels by one position
   - For bidirectional (masked) language modeling, our `create_masked_lm_inputs` function applies the appropriate masking strategy

3. **Inference**:
   - The `encode` method converts text to token IDs for model input
   - The `decode` method converts model output back to readable text

## 5. Performance Considerations

Our implementation balances functionality with performance:

1. **Hugging Face Tokenizers Library**:
   - We use the Rust-backed tokenizers library for high-performance tokenization
   - The Rust implementation is significantly faster than pure Python alternatives

2. **Batching**:
   - We support batched operations for both encoding and decoding
   - This is crucial for efficient model training and inference

3. **Padding and Truncation**:
   - We handle padding and truncation in a vectorized way where possible
   - This avoids inefficient item-by-item processing

## 6. Tokenization Limitations and Considerations

It's important to understand some limitations and considerations:

1. **Vocabulary Size Trade-off**:
   - Larger vocabulary captures more semantics but increases model size
   - Our default of 5000 tokens balances these concerns for our small model

2. **Out-of-Vocabulary Handling**:
   - BPE handles out-of-vocabulary words by breaking them into subword units
   - Very unusual words may still be broken down to character level

3. **Context Window Implications**:
   - How text is tokenized affects how many "actual words" fit in the model's context window
   - Languages or texts with many rare words might get broken into more tokens

## 7. Conclusion

The tokenization phase is a critical component of our micro-transformer project. We've implemented a ByteLevelBPE tokenizer with all the necessary functionality for both training and inference:

- Training on custom data
- Encoding and decoding text
- Handling special tokens
- Supporting different training objectives (causal and masked LM)
- Seamless integration with PyTorch

This tokenizer will serve as the interface between raw text and our transformer model, converting between human-readable text and the numerical representations our model will work with.