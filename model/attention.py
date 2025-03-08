# model/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    This implementation uses the scaled dot-product attention.
    
    Args:
        embed_dim (int): Embedding dimension (128 in your case)
        num_heads (int): Number of attention heads (4 in your case)
        dropout (float): Dropout probability (0.1 in your case)
        
    The attention dimension per head will be embed_dim / num_heads (32 in your case)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Check if embed_dim is divisible by num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for dot product attention
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, attention_mask=None, is_causal=False):
        """
        Forward pass for multi-head attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, seq_len, embed_dim]
            key (torch.Tensor): Key tensor of shape [batch_size, seq_len, embed_dim]
            value (torch.Tensor): Value tensor of shape [batch_size, seq_len, embed_dim]
            attention_mask (torch.Tensor, optional): Mask tensor of shape [batch_size, 1, seq_len, seq_len]
                                                   or [batch_size, seq_len, seq_len]
            is_causal (bool): Whether to apply causal masking (for language modeling)
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch_size, q_len, _ = query.shape
        k_len = key.shape[1]
        v_len = value.shape[1]
        
        # Apply projections and reshape for multi-head
        q = self.q_proj(query).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, v_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # [batch_size, num_heads, q_len, head_dim] x [batch_size, num_heads, head_dim, k_len]
        # -> [batch_size, num_heads, q_len, k_len]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if requested (for language modeling)
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(q_len, k_len, device=query.device, dtype=torch.bool), 
                diagonal=1
            )
            attention_scores.masked_fill_(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for broadcasting
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        # [batch_size, num_heads, q_len, k_len] x [batch_size, num_heads, v_len, head_dim]
        # -> [batch_size, num_heads, q_len, head_dim]
        output = torch.matmul(attention_weights, v)
        
        # Reshape back to original dimensions
        # [batch_size, num_heads, q_len, head_dim] -> [batch_size, q_len, embed_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.embed_dim)
        
        # Apply output projection
        output = self.out_proj(output)
        
        return output