# ModelConfig Documentation

The documentation includes:

1. **Overview**: Introduces the ModelConfig class and its purpose in the project
2. **Class Structure**: Explains the main components of the class
3. **Core Features**: Details the key functionality:
   - Parameter management
   - Preset configurations
   - Validation
   - Serialization
   - Parameter counting
4. **Code Usage**: Provides clear examples of how to use the class
5. **Implementation Details**: Explains the internals of the class
6. **Design Patterns**: Describes the design patterns used

This documentation will help users understand how to leverage the configuration system to create different model architectures, validate parameters, and analyze model sizes. The examples show how to use the ModelConfig class both directly and through the MicroTransformer integration.

You can add this to your existing documentation to provide a complete picture of your model implementation, particularly highlighting how the configuration system enables the architectural flexibility you described in your original documentation.

Would you like me to make any changes or additions to better align with your existing documentation style?

## Overview

The `ModelConfig` class provides a flexible, parameter-driven configuration system for the Micro-Transformer architecture. It serves as a single centralized configuration manager that handles all model hyperparameters, validation, serialization, and parameter counting.

This design enables researchers and developers to:
- Create models with different architectures (wide & shallow, narrow & deep, balanced)
- Store and load model configurations for reproducibility
- Validate parameter combinations to prevent invalid configurations
- Calculate and analyze parameter count breakdowns for documentation and reporting

## File Location

```
model/
├── config.py        # Configuration management system
```

## Class Structure

The `ModelConfig` class implements:

1. **Hyperparameter Storage**: Centralized storage for all model parameters
2. **Preset Architectures**: Pre-defined configurations for common model types
3. **Parameter Validation**: Integrity checking of parameter combinations
4. **Serialization Methods**: Save and load functionality
5. **Parameter Counting**: Detailed breakdown of parameter distribution

## Core Features

### Configuration Parameters

The `ModelConfig` class manages the following parameters:

| Parameter      | Description                                        | Default    |
|----------------|----------------------------------------------------|------------|
| `vocab_size`   | Size of the vocabulary (token dictionary)          | 5000       |
| `max_seq_len`  | Maximum sequence length the model can process      | 512        |
| `embed_dim`    | Embedding dimension for tokens                     | 192        |
| `num_heads`    | Number of attention heads                          | 6          |
| `num_layers`   | Number of transformer blocks                       | 4          |
| `ff_dim`       | Feed-forward hidden dimension                      | 512        |
| `dropout`      | Dropout probability for regularization             | 0.1        |
| `tie_weights`  | Whether to tie embedding and output weights        | True       |

### Preset Configurations

The class provides four predefined configurations:

1. **Tiny**: A minimal model (~0.3M parameters)
   - Settings: `embed_dim=128, num_heads=4, num_layers=2, ff_dim=256`
   - Use case: Testing, prototyping, resource-constrained environments

2. **Balanced**: A balanced model (~2.4M parameters)
   - Settings: `embed_dim=192, num_heads=6, num_layers=4, ff_dim=512`
   - Use case: General-purpose language modeling with balanced width and depth

3. **Wide & Shallow**: A wide but shallow model (~3M parameters)
   - Settings: `embed_dim=256, num_heads=8, num_layers=3, ff_dim=768`
   - Use case: Tasks requiring rich token representations but less sequential depth

4. **Narrow & Deep**: A narrow but deep model (~2.3M parameters)
   - Settings: `embed_dim=160, num_heads=5, num_layers=6, ff_dim=480`
   - Use case: Tasks with long-range dependencies requiring deeper sequential processing

### Parameter Validation

The class performs extensive validation to ensure model integrity:

- Checks that all parameters have appropriate types
- Verifies numerical parameters are within valid ranges
- Ensures that `embed_dim` is divisible by `num_heads`
- Validates other parameter combinations and dependencies

### Serialization

The class provides complete serialization capabilities:

- `to_dict()`: Converts configuration to a dictionary
- `from_dict()`: Creates configuration from a dictionary
- `save()`: Saves configuration to a JSON file
- `load()`: Loads configuration from a JSON file

### Parameter Analysis

The class offers detailed parameter counting and analysis:

- Calculated parameter count breakdowns by component
- Estimation of model size and memory footprint
- Comparison between different configurations

## Code Usage

### Basic Initialization

```python
from model.config import ModelConfig

# Create with default parameters
config = ModelConfig()

# Create with custom parameters
custom_config = ModelConfig(
    vocab_size=10000,
    embed_dim=160, 
    num_heads=5,
    num_layers=5,
    ff_dim=640,
    dropout=0.2
)
```

### Using Presets

```python
# Create from preset
tiny_config = ModelConfig.from_preset("tiny")
balanced_config = ModelConfig.from_preset("balanced")
wide_config = ModelConfig.from_preset("wide_shallow")
deep_config = ModelConfig.from_preset("narrow_deep")
```

### Modifying Configurations

```python
# Create a new configuration with updated parameters
modified_config = config.update(
    embed_dim=256,
    num_heads=8,
    dropout=0.2
)
```

### Saving and Loading

```python
# Save configuration to file
config.save("configs/my_model_config.json")

# Load configuration from file
loaded_config = ModelConfig.load("configs/my_model_config.json")
```

### Parameter Analysis

```python
# Get a detailed parameter count breakdown
param_summary = config.get_param_count_summary()
print(param_summary)
```

### Integration with MicroTransformer

```python
from model.transformer import MicroTransformer
from model.config import ModelConfig

# Create a configuration
config = ModelConfig(embed_dim=192, num_heads=6, num_layers=4)

# Create model with configuration
model = MicroTransformer(config=config)

# Or use a preset directly
model = MicroTransformer.from_preset("balanced")

# Save model configuration
model.save_config("configs/model_config.json")
```

## Implementation Details

### Configuration Validation

The validation system ensures model integrity by checking:

1. **Type validation**: All parameters have the correct data types
2. **Range validation**: Numerical parameters are within valid ranges
3. **Combination validation**: Parameter combinations make sense together
4. **Division validation**: `embed_dim` is divisible by `num_heads`

### Parameter Counting

The parameter counting system calculates parameters for each component:

1. **Token Embedding**: `vocab_size * embed_dim`
2. **Position Embedding**: `max_seq_len * embed_dim`
3. **Attention Layers**: `4 * embed_dim * embed_dim * num_layers`
4. **Feed-forward Layers**: `(embed_dim * ff_dim + ff_dim * embed_dim) * num_layers`
5. **Layer Norms**: `2 * embed_dim * (2 * num_layers + 1)`
6. **Output Head**: `embed_dim * vocab_size` (if not weight-tied)

This enables detailed analysis of where parameters are distributed in the model.

### Update Mechanism

The update system creates new configurations to ensure immutability:

1. Convert current configuration to dictionary
2. Update dictionary with new parameters
3. Create and return a new configuration instance
4. Validate the new configuration

This prevents accidental modification of existing configurations.

## Design Patterns

The `ModelConfig` class implements several design patterns:

1. **Immutable Object**: Configurations can't be modified after creation
2. **Factory Method**: Static creation methods for different use cases
3. **Builder Pattern**: Methods for constructing modified configurations
4. **Prototype Pattern**: Create new configurations from existing ones
5. **Serialization Interface**: Standard methods for saving/loading

## Conclusion

The `ModelConfig` class serves as a complete configuration system for the Micro-Transformer architecture. By centralizing and standardizing configuration management, it enhances reproducibility, experimentation, and documentation.

This system enables the easy exploration of different architectural trade-offs while maintaining a consistent API for model creation and usage.