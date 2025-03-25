# scripts/test_trainer_checkpointing.py
import os
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import MicroTransformer
from model.config import ModelConfig
from training.trainer import Trainer
from training.checkpointing import CheckpointManager, EarlyStopping

def create_dummy_batch(batch_size, seq_len, vocab_size):
    """Create a dummy batch for testing."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

class DummyDataLoader:
    def __init__(self, batch_size, seq_len, vocab_size, num_batches=5):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_batches = num_batches
    
    def __iter__(self):
        for _ in range(self.num_batches):
            yield create_dummy_batch(self.batch_size, self.seq_len, self.vocab_size)
    
    def __len__(self):
        return self.num_batches

def test_trainer_checkpointing():
    print("Testing Trainer and Checkpointing Integration...")
    
    # Create output directory
    output_dir = Path("test_checkpoints")
    output_dir.mkdir(exist_ok=True)
    
    # Create model
    model = MicroTransformer.from_preset("tiny")
    
    # Create dummy data loaders
    batch_size, seq_len = 4, 16
    train_loader = DummyDataLoader(batch_size, seq_len, model.vocab_size, num_batches=3)
    val_loader = DummyDataLoader(batch_size, seq_len, model.vocab_size, num_batches=2)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=str(output_dir),
        config={
            "log_interval": 1,
            "eval_interval": 2,
            "save_interval": 2,
            "fp16": False
        }
    )
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir=str(output_dir / "ckpt"),
        model=model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        save_interval=2
    )
    
    # Create early stopping
    early_stopping = EarlyStopping(patience=2, min_delta=0.01)
    
    print("\n1. Testing initial model training...")
    # Train for a few steps
    trainer._train_epoch()
    
    print("\n2. Testing checkpoint saving...")
    checkpoint_path = checkpoint_manager.save(
        global_step=trainer.global_step,
        epoch=trainer.epoch,
        metrics={"val_loss": 2.5},
        is_best=True
    )
    print(f"Checkpoint saved to: {checkpoint_path}")
    
    print("\n3. Testing model parameter modification...")
    # Modify model parameters to simulate training progress
    for param in model.parameters():
        with torch.no_grad():
            param.add_(torch.randn_like(param) * 0.1)
    
    print("\n4. Testing checkpoint loading...")
    checkpoint = checkpoint_manager.load(load_best=True)
    print(f"Loaded checkpoint from step {checkpoint['global_step']}")
    
    print("\n5. Testing early stopping...")
    print(f"Early stopping triggered: {early_stopping(2.5)}")  # First call
    print(f"Early stopping triggered: {early_stopping(2.52)}")  # No improvement
    print(f"Early stopping triggered: {early_stopping(2.55)}")  # No improvement, should trigger
    
    print("\n6. Testing checkpoint metadata...")
    metadata_path = output_dir / "ckpt" / "checkpoint_metadata.json"
    print(f"Metadata file exists: {metadata_path.exists()}")
    
    # Clean up test files
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_trainer_checkpointing()