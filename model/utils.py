# model/utils.py
import torch
import json
import os


def save_model_config(model, config_file):
    """
    Save model configuration to a JSON file.
    
    Args:
        model (MicroTransformer): The model instance
        config_file (str): Path to save the configuration file
    """
    config = {
        "vocab_size": model.vocab_size,
        "max_seq_len": model.max_seq_len,
        "embed_dim": model.embed_dim,
        "num_heads": model.num_heads,
        "num_layers": model.num_layers,
        "ff_dim": model.blocks[0].ff.fc1.out_features if model.blocks else None,
        "dropout": model.embedding_dropout.p
    }
    
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)


def load_model_config(config_file):
    """
    Load model configuration from a JSON file.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: Model configuration
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """
    Calculate the model size in MB.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        float: Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def create_causal_mask(seq_len, device=None):
    """
    Create a causal mask for autoregressive attention.
    
    Args:
        seq_len (int): Sequence length
        device (torch.device, optional): Device to put the mask on
        
    Returns:
        torch.Tensor: Causal mask of shape [1, 1, seq_len, seq_len]
    """
    # Create a lower triangular matrix (including diagonal)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    # Reshape for broadcasting in attention layers [1, 1, seq_len, seq_len]
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def get_sample_data(batch_size, seq_len, vocab_size, device=None):
    """
    Create sample data for testing the model.
    
    Args:
        batch_size (int): Batch size
        seq_len (int): Sequence length
        vocab_size (int): Vocabulary size
        device (torch.device, optional): Device to put the data on
        
    Returns:
        tuple: (input_ids, attention_mask)
    """
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    
    # Randomly mask some positions (to simulate padding)
    for i in range(batch_size):
        pad_len = torch.randint(0, seq_len // 4, (1,))
        if pad_len > 0:
            attention_mask[i, -pad_len:] = 0
            input_ids[i, -pad_len:] = 0  # Use 0 as padding token
    
    return input_ids, attention_mask