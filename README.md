# micro-transformer
Micro-transformer implementation trained via distributed training from scratch

```
micro-transformer/
├── README.md                  # Project overview, setup instructions, usage examples
├── requirements.txt           # All dependencies with version specifications
├── setup.py                   # Package installation script
├── .gitignore                 # Ignore patterns for checkpoints, cache, etc.
├── configs/                   # Configuration files
│   ├── model_config.json      # Model architecture parameters
│   ├── train_config.json      # Training hyperparameters
│   └── distributed_config.json # Distributed training settings
├── data/
│   ├── __init__.py
│   ├── preprocessing.py       # Data cleaning and preparation
│   ├── dataset.py             # PyTorch dataset implementations
│   └── dataloader.py          # Dataloader with batching logic
├── tokenization/
│   ├── __init__.py
│   ├── tokenizer.py           # Tokenizer implementation
│   └── utils.py               # Helper functions for tokenization
├── model/
│   ├── __init__.py
│   ├── layers.py              # Core transformer components
│   ├── attention.py           # Attention mechanism implementations
│   ├── transformer.py         # Full transformer architecture
│   └── utils.py               # Model utility functions
├── training/
│   ├── __init__.py
│   ├── trainer.py             # Main training loop
│   ├── optimizer.py           # Optimizer and scheduling setup
│   ├── checkpointing.py       # Checkpoint management
│   └── metrics.py             # Evaluation metrics
├── distributed/
│   ├── __init__.py
│   ├── data_parallel.py       # DistributedDataParallel implementation
│   ├── pipeline_parallel.py   # Pipeline parallelism implementation
│   ├── tensor_parallel.py     # Tensor parallelism implementation
│   └── utils.py               # Distributed training utilities
├── azure/
│   ├── vm_setup.sh            # VM configuration script
│   ├── distributed_setup.sh   # Multi-VM setup script
│   └── monitoring.py          # Performance monitoring tools
├── scripts/
│   ├── train.py               # Single-GPU training script
│   ├── train_ddp.py           # Data-parallel training script
│   ├── train_pipeline.py      # Pipeline-parallel training script
│   ├── train_tensor.py        # Tensor-parallel training script
│   ├── benchmark.py           # Performance benchmarking
│   └── generate.py            # Text generation using the model
└── notebooks/
    ├── data_exploration.ipynb # Dataset analysis
    ├── tokenizer_training.ipynb # Tokenizer development
    └── model_testing.ipynb    # Interactive model testing
```
