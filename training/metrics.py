# training/metrics.py
import torch
import numpy as np
from collections import defaultdict

class MetricsTracker:
    """
    Tracker for training and evaluation metrics.
    """
    def __init__(self):
        self.metrics = defaultdict(list)
        self.step_metrics = {}
    
    def update(self, metrics_dict):
        """
        Update metrics with new values.
        
        Args:
            metrics_dict: Dictionary of metric name -> value
        """
        for name, value in metrics_dict.items():
            self.metrics[name].append(value)
            self.step_metrics[name] = value
    
    def get_latest(self, name):
        """Get the most recent value for a metric."""
        if name in self.metrics and len(self.metrics[name]) > 0:
            return self.metrics[name][-1]
        return None
    
    def get_average(self, name):
        """Get the average value for a metric over all updates."""
        if name in self.metrics and len(self.metrics[name]) > 0:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return None
    
    def get_moving_average(self, name, window=100):
        """Get the moving average for a metric over the last window updates."""
        if name in self.metrics and len(self.metrics[name]) > 0:
            values = self.metrics[name][-window:]
            return sum(values) / len(values)
        return None
    
    def get_all_averages(self):
        """Get averages for all metrics."""
        return {name: self.get_average(name) for name in self.metrics}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = defaultdict(list)
        self.step_metrics = {}
    
    def get_current_values(self):
        """Get the current values for reporting."""
        return self.step_metrics

def compute_perplexity(loss):
    """
    Compute perplexity from language modeling loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        perplexity: The perplexity value
    """
    return torch.exp(torch.tensor(loss)).item()

def compute_accuracy(logits, labels, ignore_index=-100):
    """
    Compute prediction accuracy from logits and labels.
    
    Args:
        logits: Model predictions (B, S, V)
        labels: Target labels (B, S)
        ignore_index: Index to ignore in accuracy calculation
        
    Returns:
        accuracy: The prediction accuracy
    """
    # Get predictions
    preds = torch.argmax(logits, dim=-1)  # (B, S)
    
    # Create mask for valid positions
    valid_mask = (labels != ignore_index)
    
    # Count correct predictions
    correct = (preds == labels) & valid_mask
    
    # Calculate accuracy
    accuracy = correct.sum().float() / valid_mask.sum().float()
    
    return accuracy.item()

def compute_token_level_metrics(logits, labels, ignore_index=-100):
    """
    Compute token-level prediction metrics.
    
    Args:
        logits: Model predictions (B, S, V)
        labels: Target labels (B, S)
        ignore_index: Index to ignore in calculations
        
    Returns:
        metrics: Dictionary of token-level metrics
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Get predictions
    preds = torch.argmax(logits, dim=-1)  # (B, S)
    
    # Create mask for valid positions
    valid_mask = (labels != ignore_index)
    
    # Count correct predictions
    correct = (preds == labels) & valid_mask
    
    # Calculate metrics
    accuracy = correct.sum().float() / valid_mask.sum().float()
    
    # Extract top-k accuracy
    probs = torch.softmax(logits, dim=-1)  # (B, S, V)
    
    # Reshape for easier processing
    flat_probs = probs.view(-1, vocab_size)  # (B*S, V)
    flat_labels = labels.view(-1)  # (B*S)
    flat_mask = valid_mask.view(-1)  # (B*S)
    
    # Calculate top-k accuracy for k=5
    _, top5_indices = torch.topk(flat_probs, k=5, dim=1)
    top5_correct = torch.any(top5_indices == flat_labels.unsqueeze(1), dim=1)
    top5_accuracy = (top5_correct & flat_mask).sum().float() / flat_mask.sum().float()
    
    return {
        "accuracy": accuracy.item(),
        "top5_accuracy": top5_accuracy.item()
    }