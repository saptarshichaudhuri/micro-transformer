from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# Add the root directory to the path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import TransformerDataset

def create_dataloaders(train_path, val_path, tokenizer, batch_size=16, seq_length=128, num_workers=0):
    """
    Create DataLoader objects for training and validation.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        tokenizer: Tokenizer object
        batch_size: Batch size for training
        seq_length: Sequence length for model input
        num_workers: Number of worker processes for data loading
    """
    # Determine if we can use multiple workers
    # Only use multi-processing if available and on non-Windows systems
    # (Windows has issues with multiprocessing in PyTorch)
    if num_workers > 0 and os.name != 'nt' and mp.get_start_method() in ['spawn', 'forkserver']:
        use_workers = num_workers
    else:
        use_workers = 0
    
    # Create datasets
    train_dataset = TransformerDataset(train_path, tokenizer, seq_length)
    val_dataset = TransformerDataset(val_path, tokenizer, seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=use_workers,
        pin_memory=True,  # Speeds up host to GPU transfers
        drop_last=True,   # Drop the last incomplete batch
        persistent_workers=(use_workers > 0)  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,     # No need to shuffle validation data
        num_workers=use_workers,
        pin_memory=True,
        drop_last=False,   # Keep all validation samples
        persistent_workers=(use_workers > 0)
    )
    
    return train_loader, val_loader

