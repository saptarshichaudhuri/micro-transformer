import os
import json
import copy
from typing import Dict, Any, Optional, Union


class ModelConfig:
    """
    Configuration class for MicroTransformer model.
    
    This class handles all hyperparameters for the MicroTransformer architecture,
    supporting multiple model configurations (wide/shallow, narrow/deep, balanced),
    parameter validation, and serialization/deserialization.
    
    Args:
        vocab_size (int): Size of vocabulary
        max_seq_len (int): Maximum sequence length
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        ff_dim (int): Feed-forward hidden dimension
        dropout (float): Dropout probability
        tie_weights (bool): Whether to tie embedding and output layer weights
    """
    
    # Define preset configurations
    PRESETS = {
        "tiny": {
            "vocab_size": 5000,
            "max_seq_len": 512,
            "embed_dim": 128,
            "num_heads": 4,
            "num_layers": 2,
            "ff_dim": 256,
            "dropout": 0.1,
            "tie_weights": True
        },
        "balanced": {
            "vocab_size": 5000,
            "max_seq_len": 512,
            "embed_dim": 192,
            "num_heads": 6,
            "num_layers": 4,
            "ff_dim": 512,
            "dropout": 0.1,
            "tie_weights": True
        },
        "wide_shallow": {
            "vocab_size": 5000,
            "max_seq_len": 512,
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 3,
            "ff_dim": 768,
            "dropout": 0.1,
            "tie_weights": True
        },
        "narrow_deep": {
            "vocab_size": 5000,
            "max_seq_len": 512,
            "embed_dim": 160,
            "num_heads": 5,
            "num_layers": 6,
            "ff_dim": 480,
            "dropout": 0.1,
            "tie_weights": True
        }
    }
    
    def __init__(
        self,
        vocab_size: int = 5000,
        max_seq_len: int = 512,
        embed_dim: int = 192,
        num_heads: int = 6,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        # Set attributes
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.tie_weights = tie_weights
        
        # Validate parameters
        self.validate()
        
        # Calculate and store derived values
        self.head_dim = self.embed_dim // self.num_heads
        self.approx_parameters = self._calculate_parameter_count()
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'ModelConfig':
        """
        Create a configuration from a predefined preset.
        
        Args:
            preset_name (str): Name of the preset ("tiny", "balanced", "wide_shallow", "narrow_deep")
            
        Returns:
            ModelConfig: Configuration instance initialized with preset values
            
        Raises:
            ValueError: If preset_name is not recognized
        """
        if preset_name not in cls.PRESETS:
            valid_presets = ', '.join(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Valid presets are: {valid_presets}")
        
        return cls(**cls.PRESETS[preset_name])
    
    def validate(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Check vocab_size
        if not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive integer, got {self.vocab_size}")
        
        # Check max_seq_len
        if not isinstance(self.max_seq_len, int) or self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be a positive integer, got {self.max_seq_len}")
        
        # Check embed_dim
        if not isinstance(self.embed_dim, int) or self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be a positive integer, got {self.embed_dim}")
        
        # Check embed_dim is divisible by num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})")
        
        # Check num_heads
        if not isinstance(self.num_heads, int) or self.num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {self.num_heads}")
        
        # Check num_layers
        if not isinstance(self.num_layers, int) or self.num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {self.num_layers}")
        
        # Check ff_dim
        if not isinstance(self.ff_dim, int) or self.ff_dim <= 0:
            raise ValueError(f"ff_dim must be a positive integer, got {self.ff_dim}")
        
        # Check dropout
        if not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be a float between 0 and 1, got {self.dropout}")
        
        # Check tie_weights
        if not isinstance(self.tie_weights, bool):
            raise ValueError(f"tie_weights must be a boolean, got {self.tie_weights}")
    
    def _calculate_parameter_count(self) -> int:
        """
        Calculate approximate parameter count for the model.
        
        Returns:
            int: Approximate number of parameters
        """
        # Token embedding
        params = self.vocab_size * self.embed_dim
        
        # Position embedding
        params += self.max_seq_len * self.embed_dim
        
        # Each transformer block
        for _ in range(self.num_layers):
            # Attention (Q, K, V projections + output projection)
            params += 4 * (self.embed_dim * self.embed_dim)
            
            # Feed-forward (first + second layer)
            params += self.embed_dim * self.ff_dim + self.ff_dim * self.embed_dim
            
            # Layer norms (weights + biases)
            params += 4 * self.embed_dim  # 2 layer norms with weights and biases
        
        # Final layer norm
        params += 2 * self.embed_dim
        
        # LM head (if not weight-tied)
        if not self.tie_weights:
            params += self.vocab_size * self.embed_dim
        
        return params
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            dict: Configuration as a dictionary
        """
        return {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "tie_weights": self.tie_weights
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict (dict): Dictionary containing configuration parameters
            
        Returns:
            ModelConfig: Configuration instance
        """
        return cls(**config_dict)
    
    def save(self, config_file: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config_file (str): Path to save the configuration file
        """
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, config_file: str) -> 'ModelConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            config_file (str): Path to the configuration file
            
        Returns:
            ModelConfig: Configuration instance
            
        Raises:
            FileNotFoundError: If config_file does not exist
        """
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def __repr__(self) -> str:
        """
        String representation of the configuration.
        
        Returns:
            str: Human-readable configuration summary
        """
        param_str = ', '.join(f"{k}={v}" for k, v in self.to_dict().items())
        return f"ModelConfig({param_str})"
    
    def get_param_count_summary(self) -> str:
        """
        Get a summary of parameter counts for different parts of the model.
        
        Returns:
            str: Parameter count summary
        """
        # Token embedding
        token_emb_params = self.vocab_size * self.embed_dim
        
        # Position embedding
        pos_emb_params = self.max_seq_len * self.embed_dim
        
        # Attention blocks (all layers)
        attn_params = self.num_layers * 4 * (self.embed_dim * self.embed_dim)
        
        # Feed-forward blocks (all layers)
        ff_params = self.num_layers * (self.embed_dim * self.ff_dim + self.ff_dim * self.embed_dim)
        
        # Layer norms (all layers + final)
        ln_params = (2 * self.num_layers + 1) * 2 * self.embed_dim
        
        # LM head (if not weight-tied)
        lm_head_params = 0 if self.tie_weights else self.vocab_size * self.embed_dim
        
        # Total
        total = token_emb_params + pos_emb_params + attn_params + ff_params + ln_params + lm_head_params
        
        # Format output
        lines = [
            "Parameter Count Summary:",
            f"  Token Embedding:  {token_emb_params / 1e6:.2f}M",
            f"  Position Embedding: {pos_emb_params / 1e6:.2f}M",
            f"  Attention Layers:   {attn_params / 1e6:.2f}M",
            f"  Feed-forward Layers: {ff_params / 1e6:.2f}M",
            f"  Layer Norms:      {ln_params / 1e6:.2f}M",
            f"  LM Head:          {lm_head_params / 1e6:.2f}M",
            f"  --------------------------",
            f"  Total:            {total / 1e6:.2f}M parameters"
        ]
        
        return "\n".join(lines)
    
    def __eq__(self, other: Any) -> bool:
        """
        Compare two configurations for equality.
        
        Args:
            other: Object to compare with
            
        Returns:
            bool: True if configurations are equal
        """
        if not isinstance(other, ModelConfig):
            return False
        
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in ['vocab_size', 'max_seq_len', 'embed_dim', 'num_heads', 
                         'num_layers', 'ff_dim', 'dropout', 'tie_weights']
        )
    
    def update(self, **kwargs: Any) -> 'ModelConfig':
        """
        Create a new configuration with updated values.
        
        Args:
            **kwargs: Key-value pairs to update
            
        Returns:
            ModelConfig: New configuration instance with updated values
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.__class__.from_dict(config_dict)