# MicroTransformer DataLoader Documentation

## Overview

This documentation details the data loading pipeline implemented for the micro-transformer project, focusing on how text data is processed, tokenized, and served to the model during training and evaluation.

## Basic Usage

The data loading system consists of two main components:

1. `TransformerDataset`: Handles loading and tokenizing text data
2. `create_dataloaders`: Creates optimized PyTorch DataLoader instances

To use the data pipeline in your training scripts:

```python
from data.dataloader import create_dataloaders
from tokenization.tokenizer import MicroTransformerTokenizer

# Initialize tokenizer
tokenizer = MicroTransformerTokenizer.from_pretrained('path/to/tokenizer')

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    train_path='data/train.jsonl',
    val_path='data/val.jsonl',
    tokenizer=tokenizer,
    batch_size=16,
    seq_length=128
)

# Use in training loop
for batch in train_loader:
    inputs = batch['input_ids']
    labels = batch['labels']
    # Forward pass, loss calculation, etc.
```

## TransformerDataset

### Purpose
The `TransformerDataset` class extends PyTorch's `Dataset` to handle text data in JSONL or plain text format, preparing it for causal language modeling.

### Key Features

- **Flexible Data Format Support**: Handles both JSONL and plain text files
- **Tokenization Integration**: Works with the custom `MicroTransformerTokenizer`
- **Causal Language Modeling**: Generates input/target pairs by offsetting sequences
- **Memory Efficiency**: Tokenizes all data upfront to avoid redundant processing

### Implementation Details

The dataset initializes by:
1. Loading text samples from JSONL or plain text
2. Tokenizing all samples into a single continuous sequence
3. Preparing to extract fixed-length segments during iteration

During iteration, it:
1. Extracts a sequence starting at the requested index
2. Creates a target sequence shifted by one position (for next-token prediction)
3. Returns both as tensors in a dictionary format compatible with transformer models

## Core DataLoader Concepts

### Batching
**Definition**: Grouping multiple samples together to process them simultaneously, improving computational efficiency.

**Implementation**: In `dataloader.py`, the `batch_size` parameter in `create_dataloaders` determines how many samples are grouped:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,  # Controls the number of samples per batch
    # other parameters...
)
```

### Shuffling
**Definition**: Randomizing the order of samples in each epoch to reduce overfitting and improve model generalization.

**Implementation**: Enabled for training but disabled for validation:

```python
train_loader = DataLoader(
    train_dataset,
    shuffle=True,  # Randomizes sample order for training
    # other parameters...
)

val_loader = DataLoader(
    val_dataset,
    shuffle=False,  # Maintains consistent order for validation
    # other parameters...
)
```

### Prefetching
**Definition**: Loading the next batch while the current batch is being processed by the model, reducing waiting time.

**Implementation**: Implicitly handled by the `num_workers` parameter:

```python
train_loader = DataLoader(
    train_dataset,
    num_workers=use_workers,  # Controls parallel data loading
    # other parameters...
)
```

### Pin Memory
**Definition**: Allocating data in pinned (non-pageable) memory, which speeds up CPU to GPU transfers.

**Implementation**: Enabled for both training and validation loaders:

```python
train_loader = DataLoader(
    train_dataset,
    pin_memory=True,  # Speeds up host to GPU transfers
    # other parameters...
)
```

### Num Workers (Parallel Loading)
**Definition**: Using multiple CPU processes to load and preprocess data in parallel.

**Implementation**: Dynamically determined based on system compatibility:

```python
# In dataloader.py
if num_workers > 0 and os.name != 'nt' and mp.get_start_method() in ['spawn', 'forkserver']:
    use_workers = num_workers
else:
    use_workers = 0

train_loader = DataLoader(
    train_dataset,
    num_workers=use_workers,
    persistent_workers=(use_workers > 0)  # Keeps workers alive between epochs
    # other parameters...
)
```

## DataLoader Implementation

### Purpose
The `create_dataloaders` function configures optimized PyTorch DataLoader instances for both training and validation.

### Key Features

- **Performance Optimization**: Configures multiprocessing for faster data loading
- **Platform Awareness**: Adapts to different operating systems
- **Memory Efficiency**: Uses pin_memory for faster GPU transfers
- **Training-specific Settings**: Applies different configurations for training vs. validation

### Performance Optimizations

- **Persistent Workers**: Keeps worker processes alive between epochs
- **Pin Memory**: Speeds up CPU to GPU transfers
- **Dynamic Worker Count**: Adapts based on platform compatibility
- **Shuffling**: Only applied to training data, not validation
- **Drop Last**: Discards incomplete batches during training but keeps all samples during validation

## Integration with Tokenization

The data pipeline integrates with the `MicroTransformerTokenizer` which:

1. Uses ByteLevelBPE for efficient tokenization
2. Handles special tokens ([CLS], [SEP], [MASK], [PAD])
3. Returns dictionaries with input_ids, token_type_ids, and attention_mask

When the dataset receives the tokenizer output, it:
1. Extracts just the input_ids from the returned dictionary as seen in `dataset.py`:
   ```python
   # Bug 1001: Making change to the below code to read tokenizer.encode output which is a dict
   # For now only extracting the input_ids from the dict
   for sample in self.samples:
       # Use the tokenizer and extract input_ids from the returned dict
       tokens_dict = self.tokenizer.encode(sample)
       # Extract input_ids from the dictionary
       input_ids = tokens_dict["input_ids"]
       # Add the input_ids to our sequence
       self.tokenized_data.extend(input_ids)
   ```
2. Creates a continuous sequence of token IDs for efficient sampling

## Testing and Validation

### Test Script Functionality

The `testdataloader.py` script validates the data pipeline by:

1. Loading a pretrained tokenizer
2. Creating dataloaders with small test datasets
3. Checking batch shapes and contents
4. Verifying proper alignment between inputs and labels
5. Testing only a few batches to confirm functionality

### Output Validation

The script checks that:
- DataLoader produces the expected number of batches
- Tensors have the correct shapes (batch_size × seq_length)
- Input and label sequences are properly aligned for causal LM training

## Best Practices

### Performance Optimization

- **Worker Count**: Use `num_workers=cpu_count()` on Linux/Mac systems
- **Batch Size**: Adjust based on available GPU memory
- **Persistent Workers**: Enable for faster training on multi-epoch runs
- **Pin Memory**: Always enable when training on GPU

### Data Format Recommendation

- Use JSONL format for large datasets (`is_jsonl=True`)
- Structure each record with a 'text' field containing the content
- Process raw text files into JSONL during preprocessing

### Integration with Distributed Training

When using distributed training:
1. Create a separate dataloader for each GPU/process
2. Use `DistributedSampler` instead of random shuffling
3. Set appropriate settings for each parallelism type:
   - Data Parallel: Split batches across GPUs
   - Pipeline Parallel: Use the same data on each pipeline stage
   - Tensor Parallel: Use the same data on each tensor-parallel worker

## Known Issues and Solutions

- **Bug 1001**: The code now correctly handles the dictionary output from `tokenizer.encode()` by extracting just the input_ids
- **Windows Compatibility**: Multiprocessing is disabled on Windows systems
- **Memory Usage**: For very large datasets, consider implementing on-the-fly tokenization or chunked processing

## Advanced Configurations

For large-scale training, consider:

1. Implementing dataset sharding for distributed training
2. Adding support for dynamic sequence lengths based on content
3. Implementing more sophisticated chunking strategies for long documents
4. Adding data augmentation techniques during loading

## Code Examples

### Complete Example: Training a Model

```python
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model.transformer import MicroTransformer
from data.dataloader import create_dataloaders
from tokenization.tokenizer import MicroTransformerTokenizer

# Initialize tokenizer
tokenizer = MicroTransformerTokenizer.from_pretrained('path/to/tokenizer')

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    train_path='data/train.jsonl',
    val_path='data/val.jsonl',
    tokenizer=tokenizer,
    batch_size=16,
    seq_length=128,
    num_workers=4
)

# Initialize model
model = MicroTransformer(vocab_size=tokenizer.get_vocab_size())
model.to('cuda')

# Setup optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=100, 
    num_training_steps=len(train_loader) * 3  # 3 epochs
)

# Training loop
for epoch in range(3):
    model.train()
    for batch in train_loader:
        # Move batch to device
        inputs = batch['input_ids'].to('cuda')
        labels = batch['labels'].to('cuda')
        
        # Forward pass
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, tokenizer.get_vocab_size()), 
            labels.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input_ids'].to('cuda')
            labels = batch['labels'].to('cuda')
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, tokenizer.get_vocab_size()), 
                labels.view(-1)
            )
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader)}")
```