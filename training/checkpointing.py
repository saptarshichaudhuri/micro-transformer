# training/checkpointing.py
import os
import torch
import json
import time
from pathlib import Path

class CheckpointManager:
    """
    Manages model checkpoints during training.
    """
    def __init__(
        self,
        output_dir,
        model=None,
        optimizer=None,
        scheduler=None,
        save_interval=1000,
        max_checkpoints=5,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoints
            model: Model instance
            optimizer: Optimizer instance
            scheduler: Scheduler instance
            save_interval: Steps between checkpoint saves
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        
        self.saved_checkpoints = []
        self.best_metric = float("inf")
        self.global_step = 0
        self.epoch = 0
        
        # Create metadata file
        self.metadata_file = self.output_dir / "checkpoint_metadata.json"
        if not self.metadata_file.exists():
            self._save_metadata({
                "checkpoints": [],
                "best_checkpoint": None,
                "best_metric": None,
                "last_checkpoint": None,
            })
        else:
            metadata = self._load_metadata()
            self.saved_checkpoints = metadata["checkpoints"]
    
    def _save_metadata(self, metadata):
        """Save checkpoint metadata to file."""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self):
        """Load checkpoint metadata from file."""
        with open(self.metadata_file, "r") as f:
            return json.load(f)
    
    def save(self, global_step, epoch, metrics=None, is_best=False):
        """
        Save a checkpoint.
        
        Args:
            global_step: Current training step
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best checkpoint
        """
        if self.model is None:
            return
        
        # Update current step/epoch
        self.global_step = global_step
        self.epoch = epoch
        
        # Create checkpoint dict
        checkpoint = {
            "global_step": global_step,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics or {},
            "timestamp": time.time(),
        }
        
        # Add optimizer and scheduler if available
        if self.optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Create checkpoint filename and path
        checkpoint_name = f"checkpoint-{global_step}.pt"
        checkpoint_path = self.output_dir / checkpoint_name
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update saved checkpoints list
        self.saved_checkpoints.append(str(checkpoint_path))
        
        # Limit number of saved checkpoints
        if len(self.saved_checkpoints) > self.max_checkpoints:
            oldest_checkpoint = self.saved_checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
        
        # Save latest checkpoint (overwrite)
        latest_path = self.output_dir / "checkpoint-latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if applicable
        if is_best:
            best_path = self.output_dir / "checkpoint-best.pt"
            torch.save(checkpoint, best_path)
            
            # Update best metric
            if metrics and "val_loss" in metrics:
                self.best_metric = metrics["val_loss"]
        
        # Update metadata
        metadata = {
            "checkpoints": self.saved_checkpoints,
            "best_checkpoint": str(self.output_dir / "checkpoint-best.pt") if is_best else None,
            "best_metric": self.best_metric if is_best else None,
            "last_checkpoint": str(latest_path),
        }
        self._save_metadata(metadata)
        
        return checkpoint_path
    
    def load(self, checkpoint_path=None, load_best=False, load_latest=False, map_location=None):
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to specific checkpoint to load
            load_best: Whether to load the best checkpoint
            load_latest: Whether to load the latest checkpoint
            map_location: Device to map model tensors to
            
        Returns:
            checkpoint: The loaded checkpoint dict
        """
        # Determine which checkpoint to load
        if checkpoint_path is None:
            metadata = self._load_metadata()
            
            if load_best and metadata["best_checkpoint"]:
                checkpoint_path = metadata["best_checkpoint"]
            elif load_latest and metadata["last_checkpoint"]:
                checkpoint_path = metadata["last_checkpoint"]
            else:
                if len(metadata["checkpoints"]) > 0:
                    checkpoint_path = metadata["checkpoints"][-1]
                else:
                    print("No checkpoints found to load.")
                    return None
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Restore model weights if model is provided
        if self.model is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Restore optimizer if provided
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore scheduler if provided
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Update current step/epoch
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")
        
        return checkpoint
    
    def should_save(self, global_step):
        """Check if a checkpoint should be saved at this step."""
        return global_step % self.save_interval == 0

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    def __init__(self, patience=5, min_delta=0.0):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of checks with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            early_stop: Whether to stop training
        """
        score = -val_loss  # Higher score is better
        
        if self.best_score is None:
            # First check
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            # Not enough improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement found
            self.best_score = score
            self.counter = 0
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping counter."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False