import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import TransformerBlock, LayerNorm, TokenEmbedding, PositionEmbedding
from .config import ModelConfig


class MicroTransformer(nn.Module):
    """
    Micro-Transformer for language modeling.
    
    A small transformer model (~2-3M parameters) suitable for next token prediction 
    and causal language modeling tasks.
    
    Args:
        config (ModelConfig, optional): Configuration for the model. If None, default config is used.
        vocab_size (int, optional): Size of vocabulary. Overrides config if provided.
        max_seq_len (int, optional): Maximum sequence length. Overrides config if provided.
        embed_dim (int, optional): Embedding dimension. Overrides config if provided.
        num_heads (int, optional): Number of attention heads. Overrides config if provided.
        num_layers (int, optional): Number of transformer layers. Overrides config if provided.
        ff_dim (int, optional): Feed-forward hidden dimension. Overrides config if provided.
        dropout (float, optional): Dropout probability. Overrides config if provided.
        tie_weights (bool, optional): Whether to tie embedding and output weights. Overrides config if provided.
    """
    def __init__(
        self,
        config=None,
        vocab_size=None,
        max_seq_len=None,
        embed_dim=None,
        num_heads=None,
        num_layers=None,
        ff_dim=None,
        dropout=None,
        tie_weights=None
    ):
        super().__init__()
        
        # Use provided config or create default
        if config is None:
            config = ModelConfig()
        
        # Override config with any explicitly provided parameters
        override_params = {}
        if vocab_size is not None: override_params['vocab_size'] = vocab_size
        if max_seq_len is not None: override_params['max_seq_len'] = max_seq_len
        if embed_dim is not None: override_params['embed_dim'] = embed_dim
        if num_heads is not None: override_params['num_heads'] = num_heads
        if num_layers is not None: override_params['num_layers'] = num_layers
        if ff_dim is not None: override_params['ff_dim'] = ff_dim
        if dropout is not None: override_params['dropout'] = dropout
        if tie_weights is not None: override_params['tie_weights'] = tie_weights
        
        if override_params:
            config = config.update(**override_params)
        
        # Store configuration
        self.config = config
        
        # Extract parameters from config for convenience
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        
        # Embeddings
        self.token_embedding = TokenEmbedding(config.vocab_size, config.embed_dim)
        self.position_embedding = PositionEmbedding(config.max_seq_len, config.embed_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.ff_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.norm = LayerNorm(config.embed_dim)
        
        # Output projection (next token prediction head)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Tie weights between token embedding and output projection if specified
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.embedding.weight
        
        # Initialize parameters
        self.apply(self._init_weights)
        
        # Calculate and print number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f"MicroTransformer initialized with {num_params / 1_000_000:.2f}M parameters")
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            # Use Kaiming initialization for linear layers
            nn.init.kaiming_normal_(module.weight, a=0.02, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Use normal distribution for embeddings
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            # Initialize layer norm weights to 1, bias to 0
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Token IDs of shape [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len]
                where 1 indicates tokens to attend to and 0 indicates tokens to ignore
                
        Returns:
            torch.Tensor: Logits for next token prediction of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Apply embeddings
        token_emb = self.token_embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        pos_emb = self.position_embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        x = token_emb + pos_emb
        x = self.embedding_dropout(x)
        
        # Convert attention mask to a format suitable for attention layers
        # Original mask: [batch_size, seq_len] with 1s for tokens to attend to, 0s for padding
        if attention_mask is not None:
            # Convert to [batch_size, 1, 1, seq_len] and convert 0s to -inf, 1s to 0
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None
        
        # Forward pass through transformer blocks
        for block in self.blocks:
            x = block(x, extended_mask, is_causal=True)
        
        # Apply final layer norm
        x = self.norm(x)
        
        # Apply language modeling head
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text using the model.
        
        Args:
            input_ids (torch.Tensor): Starting token IDs of shape [batch_size, seq_len]
            max_length (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (1.0 means normal, <1.0 means more conservative)
            top_k (int, optional): Number of highest probability tokens to keep for sampling
            top_p (float, optional): Cumulative probability threshold for nucleus sampling
            
        Returns:
            torch.Tensor: Generated token IDs of shape [batch_size, seq_len + generated_len]
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        current_length = input_ids.shape[1]
        
        # Ensure we don't exceed max sequence length
        max_length = min(max_length, self.max_seq_len - current_length)
        
        # Create a copy of input_ids that we'll extend with generated tokens
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Only consider the last self.max_seq_len tokens
                if generated.shape[1] > self.max_seq_len:
                    inputs = generated[:, -self.max_seq_len:]
                else:
                    inputs = generated
                
                # Forward pass to get next token logits
                logits = self(inputs)
                
                # Get logits for the next token (last position)
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = torch.topk(next_token_logits, top_k, dim=-1)[0][:, -1].unsqueeze(-1)
                    next_token_logits[next_token_logits < indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i, indices_to_remove] = -float('Inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append sampled token to generated sequence
                generated = torch.cat((generated, next_token), dim=1)
                
        return generated
    
    @classmethod
    def from_config(cls, config):
        """
        Create a MicroTransformer model from a configuration.
        
        Args:
            config (ModelConfig or dict): Model configuration
            
        Returns:
            MicroTransformer: Model initialized with the given configuration
        """
        if not isinstance(config, ModelConfig):
            config = ModelConfig.from_dict(config)
        
        return cls(config=config)
    
    @classmethod
    def from_preset(cls, preset_name):
        """
        Create a MicroTransformer model from a preset configuration.
        
        Args:
            preset_name (str): Name of the preset configuration
            
        Returns:
            MicroTransformer: Model initialized with the preset configuration
        """
        config = ModelConfig.from_preset(preset_name)
        return cls(config=config)
    
    def save_config(self, config_file):
        """
        Save the model configuration to a file.
        
        Args:
            config_file (str): Path to save the configuration
        """
        self.config.save(config_file)
    
    @classmethod
    def load_config(cls, config_file):
        """
        Load a model configuration from a file.
        
        Args:
            config_file (str): Path to the configuration file
            
        Returns:
            ModelConfig: Loaded configuration
        """
        return ModelConfig.load(config_file)
    
    def get_config_summary(self):
        """
        Get a string summary of the model configuration.
        
        Returns:
            str: Configuration summary
        """
        return self.config.get_param_count_summary()