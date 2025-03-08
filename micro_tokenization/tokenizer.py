from pathlib import Path
from typing import List, Dict, Union, Optional

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing

class MicroTransformerTokenizer:
    """
    Tokenizer for the MicroTransformer model based on ByteLevelBPE.
    """
    
    def __init__(
        self, 
        vocab_file: Optional[Union[str, Path]] = None, 
        merges_file: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_file: Path to vocabulary file (if loading pretrained tokenizer)
            merges_file: Path to merges file (if loading pretrained tokenizer)
        """
        if vocab_file is not None and merges_file is not None:
            # Load pretrained tokenizer
            self.tokenizer = ByteLevelBPETokenizer(
                vocab=str(vocab_file),
                merges=str(merges_file)
            )
            # Ensure the special tokens are set
            self._add_special_tokens()
        else:
            # Initialize a new tokenizer
            self.tokenizer = ByteLevelBPETokenizer()
    
    def _add_special_tokens(self):
        """Add special tokens to the tokenizer."""
        self.tokenizer.add_special_tokens([
            "[CLS]",  # Start of sentence token
            "[SEP]",  # End of sentence token
            "[MASK]", # Mask token for masked language modeling
            "[PAD]"   # Padding token
        ])
        
        # Add post-processor for adding special tokens during encoding
        self.tokenizer.post_processor = RobertaProcessing(
            sep=("[SEP]", self.tokenizer.token_to_id("[SEP]")),
            cls=("[CLS]", self.tokenizer.token_to_id("[CLS]")),
        )
    
    def train(
        self, 
        files: List[Union[str, Path]], 
        vocab_size: int = 5000, 
        min_frequency: int = 2,
        show_progress: bool = True
    ):
        """
        Train the tokenizer on the given files.
        
        Args:
            files: List of files to train on
            vocab_size: Size of the vocabulary
            min_frequency: Minimum frequency for a token to be included
            show_progress: Whether to show a progress bar during training
        """
        # Convert all paths to strings
        files = [str(f) for f in files]
        
        # Train the tokenizer
        self.tokenizer.train(
            files=files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=[
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "[PAD]"
            ]
        )
        
        # Add post-processor
        self._add_special_tokens()
    
    def save(self, directory: Union[str, Path]):
        """
        Save the tokenizer files to the specified directory.
        
        Args:
            directory: Directory to save the tokenizer files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save the tokenizer files
        self.tokenizer.save_model(str(directory))
        
        # Also save config info
        config = {
            "vocab_size": self.tokenizer.get_vocab_size(),
            "special_tokens": {
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
                "pad_token": "[PAD]"
            }
        }
        
        import json
        with open(directory / "tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, directory: Union[str, Path]):
        """
        Load a pretrained tokenizer from a directory.
        
        Args:
            directory: Directory containing the tokenizer files
        
        Returns:
            MicroTransformerTokenizer: Loaded tokenizer
        """
        directory = Path(directory)
        vocab_file = directory / "vocab.json"
        merges_file = directory / "merges.txt"
        
        if not vocab_file.exists() or not merges_file.exists():
            raise ValueError(f"Tokenizer files not found in {directory}")
        
        return cls(vocab_file=vocab_file, merges_file=merges_file)
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
        truncation: bool = False, 
        max_length: Optional[int] = None,
        padding: bool = False,
        return_tensors: Optional[str] = None
    ) -> Dict:
        """
        Encode a text into token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens
            truncation: Whether to truncate to max_length
            max_length: Maximum length of the encoded sequence
            padding: Whether to pad to max_length
            return_tensors: If set to 'pt', will return PyTorch tensors
        
        Returns:
            Dict containing 'input_ids', 'attention_mask', etc.
        """
        # Encode the text
        encoding = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens
        )
        
        # Extract values from the encoding object
        input_ids = list(encoding.ids)
        token_type_ids = list(encoding.type_ids)
        attention_mask = list(encoding.attention_mask)
        
        # Handle truncation
        if truncation and max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            token_type_ids = token_type_ids[:max_length]
            attention_mask = attention_mask[:max_length]
        
        # Handle padding
        if padding and max_length is not None:
            pad_id = self.tokenizer.token_to_id("[PAD]")
            pad_len = max_length - len(input_ids)
            
            if pad_len > 0:
                input_ids.extend([pad_id] * pad_len)
                token_type_ids.extend([0] * pad_len)
                attention_mask.extend([0] * pad_len)
        
        # Create result dictionary
        result = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            for key in result:
                result[key] = torch.tensor([result[key]])
        
        return result
    
    def encode_batch(
        self, 
        texts: List[str], 
        add_special_tokens: bool = True,
        truncation: bool = False, 
        max_length: Optional[int] = None,
        padding: bool = False,
        return_tensors: Optional[str] = None
    ) -> Dict:
        """
        Encode a batch of texts into token IDs.
        
        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens
            truncation: Whether to truncate to max_length
            max_length: Maximum length of the encoded sequences
            padding: Whether to pad sequences to max_length or the longest sequence
            return_tensors: If set to 'pt', will return PyTorch tensors
        
        Returns:
            Dict containing batched 'input_ids', 'attention_mask', etc.
        """
        # Encode all texts
        encodings = self.tokenizer.encode_batch(
            texts,
            add_special_tokens=add_special_tokens
        )
        
        # Extract values from the encoding objects
        all_input_ids = [list(enc.ids) for enc in encodings]
        all_token_type_ids = [list(enc.type_ids) for enc in encodings]
        all_attention_masks = [list(enc.attention_mask) for enc in encodings]
        
        # Handle truncation
        if truncation and max_length is not None:
            for i in range(len(all_input_ids)):
                if len(all_input_ids[i]) > max_length:
                    all_input_ids[i] = all_input_ids[i][:max_length]
                    all_token_type_ids[i] = all_token_type_ids[i][:max_length]
                    all_attention_masks[i] = all_attention_masks[i][:max_length]
        
        # Handle padding
        if padding:
            # Determine padding length
            pad_to_length = max_length
            if pad_to_length is None:
                pad_to_length = max(len(ids) for ids in all_input_ids)
            
            pad_id = self.tokenizer.token_to_id("[PAD]")
            
            # Apply padding to make all sequences the same length
            for i in range(len(all_input_ids)):
                pad_len = pad_to_length - len(all_input_ids[i])
                if pad_len > 0:
                    all_input_ids[i].extend([pad_id] * pad_len)
                    all_token_type_ids[i].extend([0] * pad_len)
                    all_attention_masks[i].extend([0] * pad_len)
        
        # Create result dictionary
        result = {
            "input_ids": all_input_ids,
            "token_type_ids": all_token_type_ids,
            "attention_mask": all_attention_masks
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            for key in result:
                result[key] = torch.tensor(result[key])
        
        return result
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in the decoded text
        
        Returns:
            str: Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def decode_batch(self, batch_token_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token IDs back to texts.
        
        Args:
            batch_token_ids: List of lists of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in the decoded texts
        
        Returns:
            List[str]: List of decoded texts
        """
        return self.tokenizer.decode_batch(batch_token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            int: Vocabulary size
        """
        return self.tokenizer.get_vocab_size()
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Get the vocabulary mapping.
        
        Returns:
            Dict[str, int]: Mapping from tokens to IDs
        """
        return self.tokenizer.get_vocab()
    
    def token_to_id(self, token: str) -> int:
        """
        Convert a token to its ID.
        
        Args:
            token: Token string
        
        Returns:
            int: Token ID
        """
        return self.tokenizer.token_to_id(token)
    
    def id_to_token(self, id: int) -> str:
        """
        Convert an ID to its token.
        
        Args:
            id: Token ID
        
        Returns:
            str: Token string
        """
        return self.tokenizer.id_to_token(id)