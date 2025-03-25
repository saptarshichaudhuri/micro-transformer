# scripts/profile_training.py
import os
import sys
import time
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import MicroTransformer
from model.config import ModelConfig
from data import dataloader
from micro_tokenization.tokenizer import MicroTransformerTokenizer

def profile_memory_usage(model, input_ids, attention_mask=None, labels=None, device="cuda"):
    """Profile memory usage during forward and backward pass."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    # Forward pass
    forward_start = time.time()
    outputs = model(input_ids, attention_mask)
    if device == "cuda":
        torch.cuda.synchronize()
    forward_end = time.time()
    
    # Calculate memory after forward pass
    forward_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0  # MB
    
    # Backward pass
    if labels is not None:
        loss_fn = torch.nn.CrossEntropyLoss()
        batch_size, seq_len, vocab_size = outputs.shape
        outputs = outputs.reshape(-1, vocab_size)
        labels = labels.reshape(-1)
        
        loss = loss_fn(outputs, labels)
        
        backward_start = time.time()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        backward_end = time.time()
    
        # Calculate memory after backward pass
        backward_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0  # MB
    else:
        backward_memory = 0
        backward_start = 0
        backward_end = 0
    
    return {
        "forward_time": forward_end - forward_start,
        "forward_memory_mb": forward_memory,
        "backward_time": backward_end - backward_start,
        "backward_memory_mb": backward_memory,
        "total_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0
    }

# Modified run_throughput_test function
def run_throughput_test(model, dataloader, num_steps=10, fp16=False, device="cuda"):
    """Run a throughput test to measure tokens per second."""
    model.to(device)
    model.train()
    
    # Setup mixed precision if requested
    from contextlib import nullcontext
    if fp16 and device == "cuda":
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        ctx_manager = autocast()
    else:
        ctx_manager = nullcontext()
        scaler = None
    
    total_tokens = 0
    total_time = 0
    memory_stats = []
    
    # Get an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    print(f"Running throughput test with {'FP16' if fp16 and device == 'cuda' else 'FP32'} precision...")
    
    for step, batch in enumerate(tqdm(dataloader, total=num_steps)):
        if step >= num_steps:
            break
        
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
        labels = batch["labels"].to(device)
        
        # Count tokens in this batch
        batch_tokens = input_ids.numel()
        total_tokens += batch_tokens
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Synchronize if CUDA is available
        if device == "cuda":
            torch.cuda.synchronize()
        step_start = time.time()
        
        # Forward and backward pass
        if fp16 and device == "cuda":
            with ctx_manager:
                outputs = model(input_ids, attention_mask)
                # Compute loss
                loss_fn = torch.nn.CrossEntropyLoss()
                batch_size, seq_len, vocab_size = outputs.shape
                outputs = outputs.reshape(-1, vocab_size)
                labels = labels.reshape(-1)
                loss = loss_fn(outputs, labels)
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask)
            # Compute loss
            loss_fn = torch.nn.CrossEntropyLoss()
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.reshape(-1, vocab_size)
            labels = labels.reshape(-1)
            loss = loss_fn(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
        
        if device == "cuda":
            torch.cuda.synchronize()
        step_end = time.time()
        step_time = step_end - step_start
        total_time += step_time
        
        # Collect memory stats
        if device == "cuda":
            memory_stats.append({
                "step": step,
                "memory_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
                "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2)
            })
        else:
            memory_stats.append({
                "step": step,
                "memory_allocated_mb": 0,  # Not available on CPU
                "max_memory_allocated_mb": 0  # Not available on CPU
            })
        
        # Log progress
        tokens_per_sec = batch_tokens / step_time
        mem_text = f", Memory: {memory_stats[-1]['memory_allocated_mb']:.2f} MB" if device == "cuda" else ""
        print(f"Step {step}: {tokens_per_sec:.2f} tokens/sec{mem_text}")
    
    # Calculate overall statistics
    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    max_memory = max([stats["max_memory_allocated_mb"] for stats in memory_stats]) if memory_stats and device == "cuda" else 0
    
    return {
        "tokens_per_sec": avg_tokens_per_sec,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "max_memory_mb": max_memory,
        "memory_stats": memory_stats
    }

def main():
    parser = argparse.ArgumentParser(description="Profile MicroTransformer training performance")
    parser.add_argument("--model_preset", type=str, default="tiny", help="Model preset configuration")
    parser.add_argument("--data_dir", type=str, default="data/processed/small", help="Data directory")
    parser.add_argument("--tokenizer_path", type=str, default="micro_tokenization/pretrained", help="Tokenizer path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for profiling")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for profiling")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps for profiling")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (FP16)")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    args = parser.parse_args()
    
    # Set device
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    
    print(f"Running on {device}")
    print(f"Torch version: {torch.__version__}")
    if device == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load model
    print("Creating model...")
    model = MicroTransformer.from_preset(args.model_preset)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} parameters")
    
    # Load tokenizer
    tokenizer = MicroTransformerTokenizer.from_pretrained(args.tokenizer_path)
    
    # Create dataloader
    try:
        train_path = os.path.join(args.data_dir, "train.jsonl")
        if not os.path.exists(train_path):
            print(f"Train data not found at {train_path}, creating dummy data")
            # Create dummy dataloader
            batch_size, seq_len = args.batch_size, args.seq_length
            
            class DummyDataset(torch.utils.data.Dataset):
                def __init__(self, size=100):
                    self.size = size
                
                def __len__(self):
                    return self.size
                
                def __getitem__(self, idx):
                    return {
                        "input_ids": torch.randint(0, model.vocab_size, (seq_len,)),
                        "attention_mask": torch.ones(seq_len),
                        "labels": torch.randint(0, model.vocab_size, (seq_len,))
                    }
            
            dummy_dataset = DummyDataset()
            train_loader = torch.utils.data.DataLoader(
                dummy_dataset, batch_size=batch_size, shuffle=True
            )
        else:
            # Create actual dataloader
            from data import dataloader as data_loader_module
            train_path = os.path.join(args.data_dir, "train.jsonl")
            val_path = os.path.join(args.data_dir, "validation.jsonl")
            
            train_loader, _ = data_loader_module.create_dataloaders(
                train_path=train_path,
                val_path=val_path,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                num_workers=0  # Avoid dataloader workers for profiling
            )
    except Exception as e:
        print(f"Error setting up dataloader: {e}")
        print("Creating dummy dataloader instead")
        # Create dummy dataloader
        batch_size, seq_len = args.batch_size, args.seq_length
        
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, model.vocab_size, (seq_len,)),
                    "attention_mask": torch.ones(seq_len),
                    "labels": torch.randint(0, model.vocab_size, (seq_len,))
                }
        
        dummy_dataset = DummyDataset()
        train_loader = torch.utils.data.DataLoader(
            dummy_dataset, batch_size=batch_size, shuffle=True
        )
    
    # Profile throughput
    throughput_results = run_throughput_test(
        model=model,
        dataloader=train_loader,
        num_steps=args.steps,
        fp16=args.fp16,
        device=device
    )
    
    # Print results
    print("\n--- PERFORMANCE RESULTS ---")
    print(f"Average throughput: {throughput_results['tokens_per_sec']:.2f} tokens/sec")
    print(f"Total tokens processed: {throughput_results['total_tokens']:,}")
    print(f"Total time: {throughput_results['total_time']:.2f} seconds")
    print(f"Maximum memory usage: {throughput_results['max_memory_mb']:.2f} MB")
    
    # Calculate theoretical training time for full dataset
    tokens_per_epoch = args.batch_size * args.seq_length * len(train_loader)
    time_per_epoch = tokens_per_epoch / throughput_results['tokens_per_sec']
    print(f"\nEstimated time for one epoch: {time_per_epoch:.2f} seconds "
          f"({time_per_epoch/60:.2f} minutes, {time_per_epoch/3600:.2f} hours)")

if __name__ == "__main__":
    main()

## Run instructions
"""
Run it with:

python scripts/profile_trainer_mp.py --fp16 --batch_size 32 --steps 5

For testing both FP16 and FP32 performance, run it twice (with and without --fp16).
"""