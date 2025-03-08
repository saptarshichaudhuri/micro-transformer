import os
import sys
import unittest
import torch

# Add parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import MicroTransformer


class TestModelConfig(unittest.TestCase):
    """Test cases for the ModelConfig class and its integration with MicroTransformer."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for config files
        os.makedirs("temp_configs", exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary files/folders created during testing
        if os.path.exists("temp_configs"):
            for filename in os.listdir("temp_configs"):
                file_path = os.path.join("temp_configs", filename)
                os.remove(file_path)
            os.rmdir("temp_configs")
    
    def test_default_config(self):
        """Test creation of default configuration."""
        config = ModelConfig()
        self.assertEqual(config.vocab_size, 5000)
        self.assertEqual(config.max_seq_len, 512)
        self.assertEqual(config.embed_dim, 192)
        self.assertEqual(config.num_heads, 6)
        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.ff_dim, 512)
        self.assertEqual(config.dropout, 0.1)
        self.assertTrue(config.tie_weights)
    
    def test_preset_configs(self):
        """Test preset configurations."""
        # Test tiny preset
        tiny_config = ModelConfig.from_preset("tiny")
        self.assertEqual(tiny_config.embed_dim, 128)
        self.assertEqual(tiny_config.num_layers, 2)
        
        # Test balanced preset
        balanced_config = ModelConfig.from_preset("balanced")
        self.assertEqual(balanced_config.embed_dim, 192)
        self.assertEqual(balanced_config.num_heads, 6)
        self.assertEqual(balanced_config.num_layers, 4)
        
        # Test wide_shallow preset
        wide_config = ModelConfig.from_preset("wide_shallow")
        self.assertEqual(wide_config.embed_dim, 256)
        self.assertEqual(wide_config.num_heads, 8)
        self.assertEqual(wide_config.num_layers, 3)
        
        # Test narrow_deep preset
        deep_config = ModelConfig.from_preset("narrow_deep")
        self.assertEqual(deep_config.embed_dim, 160)
        self.assertEqual(deep_config.num_heads, 5)
        self.assertEqual(deep_config.num_layers, 6)
        
        # Test invalid preset
        with self.assertRaises(ValueError):
            ModelConfig.from_preset("nonexistent_preset")
    
    def test_validation(self):
        """Test parameter validation."""
        # Test embed_dim divisible by num_heads
        with self.assertRaises(ValueError):
            ModelConfig(embed_dim=100, num_heads=3)  # 100 is not divisible by 3
        
        # Test negative values
        with self.assertRaises(ValueError):
            ModelConfig(dropout=-0.1)
        
        # Test dropout > 1
        with self.assertRaises(ValueError):
            ModelConfig(dropout=1.5)
        
        # Test invalid types
        with self.assertRaises(ValueError):
            ModelConfig(vocab_size="string")
        
        with self.assertRaises(ValueError):
            ModelConfig(tie_weights="not_a_bool")
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        original_config = ModelConfig(
            vocab_size=10000,
            max_seq_len=1024,
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            ff_dim=1024,
            dropout=0.2,
            tie_weights=False
        )
        
        # Test to_dict and from_dict
        config_dict = original_config.to_dict()
        loaded_config = ModelConfig.from_dict(config_dict)
        self.assertEqual(original_config, loaded_config)
        
        # Test save and load
        config_path = os.path.join("temp_configs", "test_config.json")
        original_config.save(config_path)
        loaded_config = ModelConfig.load(config_path)
        self.assertEqual(original_config, loaded_config)
    
    def test_update(self):
        """Test configuration update."""
        original_config = ModelConfig()
        updated_config = original_config.update(
            embed_dim=256,
            num_heads=8,
            dropout=0.2
        )
        
        # Check updated values
        self.assertEqual(updated_config.embed_dim, 256)
        self.assertEqual(updated_config.num_heads, 8)
        self.assertEqual(updated_config.dropout, 0.2)
        
        # Check unchanged values
        self.assertEqual(updated_config.vocab_size, original_config.vocab_size)
        self.assertEqual(updated_config.num_layers, original_config.num_layers)
        
        # Check that original config is not modified
        self.assertEqual(original_config.embed_dim, 192)
        self.assertEqual(original_config.num_heads, 6)
        
        # Test updating with invalid values
        with self.assertRaises(ValueError):
            original_config.update(embed_dim=101, num_heads=3)  # Not divisible
    
    def test_parameter_count(self):
        """Test parameter count calculation."""
        config = ModelConfig(
            vocab_size=5000,
            max_seq_len=512,
            embed_dim=192,
            num_heads=6,
            num_layers=4,
            ff_dim=512
        )
        
        # Get the summary
        summary = config.get_param_count_summary()
        
        # Check that the summary contains expected sections
        self.assertIn("Token Embedding", summary)
        self.assertIn("Position Embedding", summary)
        self.assertIn("Attention Layers", summary)
        self.assertIn("Feed-forward Layers", summary)
        self.assertIn("Total:", summary)
        
        # Test parameter count with different weight tying
        config_tied = ModelConfig(tie_weights=True)
        config_untied = ModelConfig(tie_weights=False)
        
        # Untied should have more parameters
        self.assertGreater(config_untied._calculate_parameter_count(), 
                           config_tied._calculate_parameter_count())
    
    def test_model_integration(self):
        """Test integration with MicroTransformer."""
        # Create model with default config
        default_model = MicroTransformer()
        self.assertEqual(default_model.embed_dim, 192)
        self.assertEqual(default_model.num_heads, 6)
        
        # Create model with custom config
        custom_config = ModelConfig(
            vocab_size=10000,
            embed_dim=160,
            num_heads=5,
            num_layers=3
        )
        custom_model = MicroTransformer(config=custom_config)
        self.assertEqual(custom_model.vocab_size, 10000)
        self.assertEqual(custom_model.embed_dim, 160)
        self.assertEqual(custom_model.num_heads, 5)
        self.assertEqual(custom_model.num_layers, 3)
        self.assertEqual(len(custom_model.blocks), 3)
        
        # Test from_preset
        tiny_model = MicroTransformer.from_preset("tiny")
        self.assertEqual(tiny_model.embed_dim, 128)
        self.assertEqual(tiny_model.num_layers, 2)
        self.assertEqual(len(tiny_model.blocks), 2)
        
        # Test parameter override
        override_model = MicroTransformer(
            config=custom_config,
            embed_dim=192,
            num_heads=6
        )
        self.assertEqual(override_model.embed_dim, 192)
        self.assertEqual(override_model.num_heads, 6)
        self.assertEqual(override_model.vocab_size, 10000)  # From config
        
        # Test save and load config through model
        config_path = os.path.join("temp_configs", "model_config.json")
        custom_model.save_config(config_path)
        loaded_config = MicroTransformer.load_config(config_path)
        self.assertEqual(custom_model.config, loaded_config)
        
        # Test model creation from loaded config
        new_model = MicroTransformer.from_config(loaded_config)
        self.assertEqual(new_model.vocab_size, custom_model.vocab_size)
        self.assertEqual(new_model.embed_dim, custom_model.embed_dim)
        self.assertEqual(new_model.num_heads, custom_model.num_heads)
    
    def test_model_functionality(self):
        """Test basic model functionality with different configs."""
        # Small batch for testing
        batch_size = 2
        seq_len = 16
        
        # Test with tiny model
        tiny_model = MicroTransformer.from_preset("tiny")
        input_ids = torch.randint(0, tiny_model.vocab_size, (batch_size, seq_len))
        output = tiny_model(input_ids)
        self.assertEqual(output.shape, (batch_size, seq_len, tiny_model.vocab_size))
        
        # Test with custom model
        custom_config = ModelConfig(
            vocab_size=1000,
            max_seq_len=128,
            embed_dim=96,
            num_heads=3,
            num_layers=2,
            ff_dim=256
        )
        custom_model = MicroTransformer(config=custom_config)
        input_ids = torch.randint(0, custom_model.vocab_size, (batch_size, seq_len))
        output = custom_model(input_ids)
        self.assertEqual(output.shape, (batch_size, seq_len, custom_model.vocab_size))
        
        # Test attention mask
        attention_mask = torch.ones((batch_size, seq_len))
        attention_mask[:, -4:] = 0  # Mask last 4 tokens
        output_with_mask = custom_model(input_ids, attention_mask)
        self.assertEqual(output_with_mask.shape, (batch_size, seq_len, custom_model.vocab_size))
        
        # Test generation
        generated = custom_model.generate(input_ids, max_length=4)
        self.assertEqual(generated.shape, (batch_size, seq_len + 4))
        
        # Test generation with different parameters
        generated_temp = custom_model.generate(input_ids, max_length=4, temperature=0.7)
        self.assertEqual(generated_temp.shape, (batch_size, seq_len + 4))
        
        generated_topk = custom_model.generate(input_ids, max_length=4, top_k=5)
        self.assertEqual(generated_topk.shape, (batch_size, seq_len + 4))
        
        generated_topp = custom_model.generate(input_ids, max_length=4, top_p=0.9)
        self.assertEqual(generated_topp.shape, (batch_size, seq_len + 4))


if __name__ == "__main__":
    unittest.main()