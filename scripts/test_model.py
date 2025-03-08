# scripts/test_model.py
import sys
import os
import torch

# Add the parent directory to the path to import the model package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import MicroTransformer
from model.utils import count_parameters, get_model_size_mb, get_sample_data

# Modified part of scripts/test_model.py
def test_configuration(config):
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
    print("Testing different MicroTransformer configurations to find one in the 2-3M parameter range...\n")
    
    # Test different configurations
    configurations = [
        {
            "vocab_size": 5000,
            "max_seq_len": 512,
            "embed_dim": 192,  # Increased from 128
            "num_heads": 6,    # Increased from 4
            "num_layers": 4,
            "ff_dim": 512,
            "dropout": 0.1
        },
        {
            "vocab_size": 5000,
            "max_seq_len": 512,
            "embed_dim": 128,
            "num_heads": 4,
            "num_layers": 6,   # Increased from 4
            "ff_dim": 768,     # Increased from 512
            "dropout": 0.1
        },
        {
            "vocab_size": 5000,
            "max_seq_len": 512,
            "embed_dim": 160,  # Increased from 128
            "num_heads": 5,    # Increased from 4
            "num_layers": 5,   # Increased from 4
            "ff_dim": 640,     # Increased from 512
            "dropout": 0.1
        }
    ]
    
    results = []
    for config in configurations:
        params, in_range = test_configuration(config)
        results.append((config, params, in_range))
    
    print("\n--- SUMMARY ---")
    for config, params, in_range in results:
        status = "✓" if in_range else "✗"
        print(f"{status} embed_dim={config['embed_dim']}, num_heads={config['num_heads']}, num_layers={config['num_layers']}, ff_dim={config['ff_dim']} => {params:,} parameters")
    
    # Find configurations in range
    valid_configs = [c for c, p, r in results if r]
    
    if valid_configs:
        print("\nRecommended configuration:")
        rec = valid_configs[0]
        print(f"embed_dim={rec['embed_dim']}, num_heads={rec['num_heads']}, num_layers={rec['num_layers']}, ff_dim={rec['ff_dim']}")
        
        # Test the model with this configuration
        model = MicroTransformer(**rec)
        
        # Create sample data and test forward pass
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