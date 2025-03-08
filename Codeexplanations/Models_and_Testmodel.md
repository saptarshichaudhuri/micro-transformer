# Micro-Transformer Implementation Documentation

The documentation explains how the model is a parameterized design that can be configured with different architectural choices while maintaining the same code structure. It highlights how the test script explores the parameter space to find configurations that meet your 2-3M parameter target.
The three configurations identified all have around 2.3-2.4M parameters but represent different architectural trade-offs:

Wide & Shallow (embed_dim=192, num_heads=6, num_layers=4): Better for complex, shorter contexts with richer token representations
Narrow & Deep (embed_dim=128, num_heads=4, num_layers=6): Better for capturing long-range dependencies with more sequential processing
Balanced (embed_dim=160, num_heads=5, num_layers=5): A middle-ground approach that balances width and depth

## 1. Code Structure

The Micro-Transformer implementation follows a modular design pattern with several key components:

### Directory Structure

```
model/
├── __init__.py         # Exports public interfaces
├── attention.py        # Attention mechanism implementation
├── layers.py           # Core transformer components
├── transformer.py      # Complete transformer architecture
└── utils.py            # Utility functions

scripts/
└── test_model.py       # Model testing and configuration exploration
```

### Component Workflow

#### Initialization Sequence

1. User instantiates a `MicroTransformer` from `transformer.py`
2. `MicroTransformer` initializes embedding layers (token and position)
3. `MicroTransformer` creates `TransformerBlock` instances
4. Each `TransformerBlock` creates a `MultiHeadAttention` and `FeedForward` layer
5. Final layer norm and prediction head are initialized

#### Forward Pass Sequence

1. `MicroTransformer.forward()` processes input token IDs
2. Token and position embeddings are applied and added
3. The sequence flows through each transformer block:
   - Attention mechanism processes the sequence
   - Feed-forward network further transforms representations
4. Final layer norm is applied
5. Prediction head converts hidden states to logits over vocabulary

#### Generation Sequence

1. `MicroTransformer.generate()` takes a prompt as input
2. For each position to generate:
   - Forward pass produces next-token logits
   - Sampling strategy selects the next token
   - The token is appended to the sequence
3. Process repeats until reaching max length or stop condition

## 2. Code Usage

### Basic Model Initialization

```python
from model.transformer import MicroTransformer

# Initialize with default parameters
model = MicroTransformer(
    vocab_size=5000,
    max_seq_len=512,
    embed_dim=192,
    num_heads=6,
    num_layers=4,
    ff_dim=512,
    dropout=0.1
)
```

### Training Loop Integration

```python
# Example of how to integrate with a training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        # Calculate loss (shift logits and labels for next-token prediction)
        # Labels are just input_ids shifted right for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = criterion(shift_logits.view(-1, model.vocab_size), shift_labels.view(-1))
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Text Generation

```python
# Generate text continuation
input_text = tokenizer.encode("Once upon a time", return_tensors="pt")
generated = model.generate(
    input_text,
    max_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
decoded_text = tokenizer.decode(generated[0])
print(decoded_text)
```

### Saving and Loading

```python
# Saving model
torch.save(model.state_dict(), "micro_transformer.pt")

# Loading model
loaded_model = MicroTransformer(vocab_size=5000, ...)
loaded_model.load_state_dict(torch.load("micro_transformer.pt"))
```

## 3. Architecture Configurations

Based on the test script results, here are three viable configurations that achieve the 2-3M parameter target:

### Configuration 1: Wide & Shallow
- **Parameters**: 2.44M
- **Settings**: embed_dim=192, num_heads=6, num_layers=4, ff_dim=512
- **Pros**:
  - Larger embedding dimension provides richer token representations
  - More attention heads can capture more diverse patterns in parallel
  - Fewer layers means faster forward/backward passes
  - Good for capturing complex token relationships in shorter contexts
- **Cons**:
  - Less sequential processing depth
  - May struggle with very long-range dependencies

### Configuration 2: Narrow & Deep
- **Parameters**: 2.29M
- **Settings**: embed_dim=128, num_heads=4, num_layers=6, ff_dim=768
- **Pros**:
  - More transformer layers allow for deeper hierarchical processing
  - Larger feed-forward networks provide more processing capacity per token
  - Better for capturing long-range dependencies and complex reasoning
  - More efficient memory usage during training
- **Cons**:
  - Smaller embeddings may limit representational capacity
  - Deeper models can be harder to train (vanishing gradients)
  - Slower inference due to sequential layer processing

### Configuration 3: Balanced
- **Parameters**: 2.43M
- **Settings**: embed_dim=160, num_heads=5, num_layers=5, ff_dim=640
- **Pros**:
  - Balanced trade-off between width and depth
  - Moderate embedding size with moderate depth
  - Good all-around performance for various tasks
  - Reasonable training and inference speed
- **Cons**:
  - Jack of all trades, master of none
  - May not excel at either short-context or very long-range tasks

### Selecting a Configuration

Choose your configuration based on your task characteristics:

- **For short, complex texts** (like code or dense technical content), the **Wide & Shallow** configuration may work best.
- **For narrative text with long dependencies** (stories, articles), the **Narrow & Deep** configuration might be more suitable.
- **For general-purpose use** across various text types, the **Balanced** configuration offers a good compromise.

## 4. Test Script Explanation

The `test_model.py` script serves two important functions:

1. **Configuration Testing**: It tries multiple model configurations to find ones that meet the parameter count target.
2. **Functionality Verification**: It tests the model's forward pass and generation capabilities.

### Key Components

- `test_configuration()`: Tests a single configuration and reports parameters
- `main()`: Tests multiple configurations and summarizes results
- Configuration dictionaries: Define different architectural options
- Model verification: Ensures the forward pass and generation work correctly

### Running the Test

```bash
python scripts/test_model.py
```

The script will output:
- Parameter counts for each tested configuration
- Which configurations meet the target range
- A summary comparison of viable options
- Verification that the recommended configuration works correctly

## 5. Implementation Details

### Attention Mechanism

The `MultiHeadAttention` class implements scaled dot-product attention with multiple heads. Key operations:

1. Project queries, keys, and values
2. Split projections into multiple heads
3. Compute scaled dot-product attention
4. Apply attention weights to values
5. Combine heads and project output

### Transformer Block

Each `TransformerBlock` uses pre-layer normalization architecture:

1. Apply layer norm
2. Process through multi-head attention
3. Add residual connection
4. Apply layer norm
5. Process through feed-forward network
6. Add residual connection

### Model Initialization

The model uses standard initialization practices:

- Linear layers: Kaiming initialization
- Embeddings: Normal distribution with small std
- Layer norms: Weights=1, biases=0

This ensures stable gradient flow during early training.

## Conclusion

This micro-transformer implementation provides a flexible, configurable architecture for language modeling tasks. By adjusting the hyperparameters, you can optimize for different task requirements while meeting your parameter budget constraints.

The modular design allows for easy extension and modification as your project evolves, particularly for the distributed training scenarios outlined in your project plan.



# Model Code Summary

# 1. model/__init__.py
# Exports public interfaces for easy imports
from .transformer import MicroTransformer
from .layers import TransformerBlock, FeedForward, LayerNorm, TokenEmbedding, PositionEmbedding
from .attention import MultiHeadAttention

__all__ = [
    'MicroTransformer',
    'TransformerBlock',
    'MultiHeadAttention', 
    'FeedForward',
    'LayerNorm',
    'TokenEmbedding',
    'PositionEmbedding'
]

# 2. model/attention.py
# Key class: MultiHeadAttention
# Implements scaled dot-product attention with multiple heads
# Flow: project Q/K/V → split heads → compute attention → combine heads → project output

# 3. model/layers.py
# Core components:
# - LayerNorm: Normalizes representations for stable training
# - TokenEmbedding: Maps token IDs to vectors, scaled by sqrt(dim)
# - PositionEmbedding: Provides position information to the model
# - FeedForward: Two-layer network with GELU activation
# - TransformerBlock: Combines attention and feedforward with residual connections

# 4. model/transformer.py
# Main model class: MicroTransformer
# Assembles all components into a complete language model
# Architecture: embeddings → transformer blocks → layer norm → prediction head
# Includes token generation functionality with sampling strategies

# 5. model/utils.py
# Utility functions:
# - save_model_config/load_model_config: Manage configuration
# - count_parameters: Tracks model size
# - get_model_size_mb: Calculates memory footprint
# - create_causal_mask: For autoregressive modeling
# - get_sample_data: For testing purposes

# 6. scripts/test_model.py
# Testing script with three key functions:
# - Test various model configurations
# - Validate model functionality (forward pass, generation)
# - Compare parameter counts and report results


# scripts/test_model.py - Full Breakdown

import sys
import os
import torch

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import MicroTransformer
from model.utils import count_parameters, get_model_size_mb, get_sample_data

def test_configuration(config):
    """
    Test a single model configuration and report its parameter count.
    
    Args:
        config (dict): Model configuration parameters
        
    Returns:
        tuple: (num_params, in_range) - parameter count and whether it's in target range
    """
    print(f"\nTesting configuration: {config}")
    
    # Create a model instance with the given configuration
    model = MicroTransformer(**config)

    # Count parameters
    num_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    print(f"Model has {num_params:,} parameters")
    print(f"Model size: {model_size:.2f} MB")
    
    # Check if in target range
    in_range = 2_000_000 <= num_params <= 3_000_000
    print(f"Parameter count in target range (2-3M): {in_range}")
    
    return num_params, in_range

def main():
    """
    Main function to test different model configurations.
    """
    print("Testing different MicroTransformer configurations to find one in the 2-3M parameter range...\n")
    
    # Test different configurations
    configurations = [
        # Configuration 1: Wide & Shallow
        {
            "vocab_size": 5000,
            "max_seq_len": 512,
            "embed_dim": 192,  # Wider embeddings (192 vs 128)
            "num_heads": 6,    # More attention heads (6 vs 4)
            "num_layers": 4,   # Standard depth
            "ff_dim": 512,     # Standard FF size
            "dropout": 0.1
        },
        # Configuration 2: Narrow & Deep
        {
            "vocab_size": 5000,
            "max_seq_len": 512,
            "embed_dim": 128,  # Standard embeddings
            "num_heads": 4,    # Standard heads
            "num_layers": 6,   # Deeper network (6 vs 4 layers)
            "ff_dim": 768,     # Larger FF networks (768 vs 512)
            "dropout": 0.1
        },
        # Configuration 3: Balanced
        {
            "vocab_size": 5000,
            "max_seq_len": 512,
            "embed_dim": 160,  # Medium embeddings
            "num_heads": 5,    # Medium heads
            "num_layers": 5,   # Medium depth
            "ff_dim": 640,     # Medium FF size
            "dropout": 0.1
        }
    ]
    
    # Test each configuration
    results = []
    for config in configurations:
        params, in_range = test_configuration(config)
        results.append((config, params, in_range))
    
    # Print summary of results
    print("\n--- SUMMARY ---")
    for config, params, in_range in results:
        status = "✓" if in_range else "✗"
        print(f"{status} embed_dim={config['embed_dim']}, num_heads={config['num_heads']}, " +
              f"num_layers={config['num_layers']}, ff_dim={config['ff_dim']} => {params:,} parameters")
    
    # Find configurations in target range
    valid_configs = [c for c, p, r in results if r]
    
    if valid_configs:
        # Present and test first valid configuration
        print("\nRecommended configuration:")
        rec = valid_configs[0]
        print(f"embed_dim={rec['embed_dim']}, num_heads={rec['num_heads']}, " +
              f"num_layers={rec['num_layers']}, ff_dim={rec['ff_dim']}")
        
        # Instantiate model with recommended config
        model = MicroTransformer(**rec)
        
        # Test forward pass
        batch_size = 4
        seq_len = 64
        input_ids, attention_mask = get_sample_data(batch_size, seq_len, model.vocab_size)
        
        print(f"\nRunning forward pass with sample data...")
        logits = model(input_ids, attention_mask)
        print(f"Output logits shape: {logits.shape}")
        
        # Test generation
        print(f"\nTesting text generation...")
        generated = model.generate(input_ids[:1, :10], max_length=20, temperature=0.8, top_k=50)
        print(f"Generated sequence shape: {generated.shape}")
        
        print("\n✓ Found a configuration in the target parameter range!")
    else:
        print("\n✗ None of the tested configurations are in the target range.")

if __name__ == "__main__":
    main()