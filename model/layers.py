# model/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """
    Layer Normalization with optional bias.
    
    Args:
        embed_dim (int): Embedding dimension
        eps (float): Small constant for numerical stability
    """
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.
    
    Args:
        vocab_size (int): Size of vocabulary (5000 for your tokenizer)
        embed_dim (int): Embedding dimension (128 in your case)
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_dim)


class PositionEmbedding(nn.Module):
    """
    Learnable position embedding layer.
    
    Args:
        max_seq_len (int): Maximum sequence length
        embed_dim (int): Embedding dimension (128 in your case)
    """
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, embed_dim)
        
    def forward(self, x):
        # Create position indices tensor
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.embedding(positions)


class FeedForward(nn.Module):
    """
    Feed-forward network used in transformer blocks.
    
    Args:
        embed_dim (int): Input/output dimension (128 in your case)
        hidden_dim (int): Hidden dimension (512 in your case)
        dropout (float): Dropout probability (0.1 in your case)
    """
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-layer normalization architecture.
    
    Args:
        embed_dim (int): Embedding dimension (128 in your case)
        num_heads (int): Number of attention heads (4 in your case)
        ff_dim (int): Feed-forward hidden dimension (512 in your case)
        dropout (float): Dropout probability (0.1 in your case)
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # Pre-attention layer norm
        self.norm1 = LayerNorm(embed_dim)
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        # Pre-feedforward layer norm
        self.norm2 = LayerNorm(embed_dim)
        # Feedforward network
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None, is_causal=False):
        # First sublayer: Multi-head attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.attention(x, x, x, attention_mask, is_causal)
        x = self.dropout(x) + residual
        
        # Second sublayer: Feed-forward network with residual connection
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x) + residual
        
        return x