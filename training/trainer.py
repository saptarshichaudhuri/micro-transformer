# training/trainer.py
import os
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        config=None,
        output_dir="checkpoints",
        device=None,
    ):
        """
        Trainer for MicroTransformer models.
        
        Args:
            model: MicroTransformer model instance
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            config: Training configuration dict
            output_dir: Directory to save checkpoints
            device: Device to train on (defaults to CUDA if available)
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Default configuration
        self.config = {
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "warmup_steps": 100,
            "max_grad_norm": 1.0,
            "fp16": True,
            "gradient_accumulation_steps": 1,
            "log_interval": 10,
            "eval_interval": 500,
            "save_interval": 1000,
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
        # Setup device
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision training
        self.fp16 = self.config["fp16"] and self.device.type == "cuda"
        self.scaler = GradScaler() if self.fp16 else None
        
        # Setup tracking variables
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        
        # Setup output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _create_optimizer(self):
        """Create and configure the AdamW optimizer."""
        # Use different weight decay for certain parameters
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config["learning_rate"],
            betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
            eps=self.config["adam_epsilon"],
        )
    
    def _create_scheduler(self):
        """Create a learning rate scheduler."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step):
            # Linear warmup followed by constant
            if current_step < self.config["warmup_steps"]:
                return float(current_step) / float(max(1, self.config["warmup_steps"]))
            return 1.0
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train(self, num_epochs):
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs
        """
        total_start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            self._train_epoch()
            
            # Evaluate at the end of each epoch
            val_loss, val_perplexity = self.evaluate()
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
            
            # Save checkpoint at epoch end
            self.save_checkpoint(is_best=val_loss < self.best_val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
        
        total_time = time.time() - total_start_time
        print(f"Training completed in {total_time:.2f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def _train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        # Progress bar
        pbar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {self.epoch+1}")
        
        for step, batch in enumerate(self.train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device) if "attention_mask" in batch else None
            labels = batch["labels"].to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.fp16:
                with autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self._compute_loss(outputs, labels)
                
                # Scale loss and backward pass
                self.scaler.scale(loss / self.config["gradient_accumulation_steps"]).backward()
                
                # Accumulate gradients
                if (step + 1) % self.config["gradient_accumulation_steps"] == 0:
                    # Clip gradients
                    if self.config["max_grad_norm"] > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config["max_grad_norm"]
                        )
                    
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.global_step += 1
            else:
                # Standard training without mixed precision
                outputs = self.model(input_ids, attention_mask)
                loss = self._compute_loss(outputs, labels)
                
                # Backward pass
                (loss / self.config["gradient_accumulation_steps"]).backward()
                
                # Accumulate gradients
                if (step + 1) % self.config["gradient_accumulation_steps"] == 0:
                    # Clip gradients
                    if self.config["max_grad_norm"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config["max_grad_norm"]
                        )
                    
                    # Update weights
                    self.optimizer.step()
                    self.scheduler.step()
                    self.global_step += 1
            
            # Update tracking variables
            epoch_loss += loss.item()
            
            # Logging
            if step % self.config["log_interval"] == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": loss.item(),
                    "lr": f"{current_lr:.2e}",
                    "step": self.global_step
                })
            
            # Evaluation
            if self.global_step % self.config["eval_interval"] == 0:
                val_loss, val_perplexity = self.evaluate()
                self.model.train()  # Switch back to train mode
                
                pbar.set_postfix({
                    "loss": loss.item(),
                    "val_loss": val_loss,
                    "val_ppl": f"{val_perplexity:.2f}",
                    "step": self.global_step
                })
                
                # Save checkpoint
                self.save_checkpoint(is_best=val_loss < self.best_val_loss)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
            
            pbar.update(1)
        
        pbar.close()
        
        # Calculate epoch stats
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(self.train_dataloader)
        
        print(f"Epoch completed in {epoch_time:.2f}s, Avg loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _compute_loss(self, outputs, labels):
        """
        Compute the loss for training.
        
        Args:
            outputs: Model outputs (logits)
            labels: Target labels
            
        Returns:
            loss: The cross-entropy loss
        """
        # Reshape outputs and labels for computing loss
        # outputs shape: (batch_size, seq_len, vocab_size)
        # labels shape: (batch_size, seq_len)
        
        batch_size, seq_len, vocab_size = outputs.shape
        outputs = outputs.reshape(-1, vocab_size)
        labels = labels.reshape(-1)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(outputs, labels, ignore_index=-100)
        
        return loss
    
    def evaluate(self):
        """
        Evaluate the model on the validation set.
        
        Returns:
            val_loss: The average validation loss
            perplexity: The perplexity (exponential of the loss)
        """
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device) if "attention_mask" in batch else None
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self._compute_loss(outputs, labels)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_dataloader)
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        
        return avg_val_loss, perplexity
    
    def generate_sample(self, prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.9):
        """
        Generate text from a prompt using the trained model.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability cutoff for nucleus sampling
            
        Returns:
            generated_text: The generated text
        """
        # This would typically rely on your tokenizer to prepare the input
        # and decode the output, which is not shown here
        
        self.model.eval()
        # Assuming you have a generate method in your model
        generated = self.model.generate(
            prompt, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p
        )
        
        return generated
    
    def save_checkpoint(self, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{self.global_step}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save latest checkpoint (overwrite)
        latest_path = os.path.join(self.output_dir, "checkpoint-latest.pt")
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint (if applicable)
        if is_best:
            best_path = os.path.join(self.output_dir, "checkpoint-best.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler if it exists
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load other training state
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")