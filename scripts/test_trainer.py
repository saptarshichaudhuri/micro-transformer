# scripts/test_trainer.py
import os
import sys
import torch
# Add the parent directory to the path to import the model package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import MicroTransformer
from model.config import ModelConfig
from micro_tokenization.tokenizer import MicroTransformerTokenizer
from data import dataloader
from training.trainer import Trainer
from training.metrics import MetricsTracker

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

def create_dummy_dataloader(batch_size, seq_len, vocab_size, num_batches=5):
    """Create a dummy dataloader with fixed number of batches."""
    class DummyDataLoader:
        def __init__(self, batch_size, seq_len, vocab_size, num_batches):
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.vocab_size = vocab_size
            self.num_batches = num_batches
        
        def __iter__(self):
            for _ in range(self.num_batches):
                yield create_dummy_batch(self.batch_size, self.seq_len, self.vocab_size)
        
        def __len__(self):
            return self.num_batches
    
    return DummyDataLoader(batch_size, seq_len, vocab_size, num_batches)

def test_trainer():
    print("Testing Trainer and Metrics components...")
    
    # Create a small model for testing
    config = ModelConfig.from_preset("tiny")
    model = MicroTransformer(config=config)
    
    # Create dummy data loaders
    batch_size = 4
    seq_len = 16
    train_loader = create_dummy_dataloader(batch_size, seq_len, config.vocab_size, num_batches=3)
    val_loader = create_dummy_dataloader(batch_size, seq_len, config.vocab_size, num_batches=2)
    
    # Configure trainer with small settings for testing
    trainer_config = {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "fp16": False,  # Disable mixed precision for testing
        "gradient_accumulation_steps": 1,
        "log_interval": 1,
        "eval_interval": 2,
        "save_interval": 3,
    }
    
    # Create output directory for checkpoints
    output_dir = "test_checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=trainer_config,
        output_dir=output_dir
    )
    
    print("\n1. Testing optimizer creation...")
    print(f"Optimizer type: {type(trainer.optimizer)}")
    print(f"Learning rate: {trainer.config['learning_rate']}")
    
    print("\n2. Testing scheduler creation...")
    print(f"Scheduler type: {type(trainer.scheduler)}")
    
    print("\n3. Testing loss computation...")
    dummy_batch = create_dummy_batch(batch_size, seq_len, config.vocab_size)
    inputs = dummy_batch["input_ids"].to(trainer.device)
    labels = dummy_batch["labels"].to(trainer.device)
    outputs = model(inputs)
    loss = trainer._compute_loss(outputs, labels)
    print(f"Computed loss: {loss.item()}")
    
    print("\n4. Testing evaluation...")
    val_loss, val_ppl = trainer.evaluate()
    print(f"Validation loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
    
    print("\n5. Testing training for 1 epoch...")
    epoch_loss = trainer._train_epoch()
    print(f"Epoch loss: {epoch_loss:.4f}")
    
    print("\n6. Testing checkpointing...")
    trainer.save_checkpoint(is_best=True)
    print(f"Checkpoint saved in {output_dir}")
    
    print("\n7. Testing MetricsTracker...")
    metrics = MetricsTracker()
    metrics.update({"loss": loss.item(), "accuracy": 0.75})
    metrics.update({"loss": loss.item() * 0.9, "accuracy": 0.78})
    print(f"Latest loss: {metrics.get_latest('loss'):.4f}")
    print(f"Average loss: {metrics.get_average('loss'):.4f}")
    print(f"All averages: {metrics.get_all_averages()}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_trainer()