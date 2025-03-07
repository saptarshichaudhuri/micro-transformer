import os
import json

# Base directory - replace with your desired path or use current directory
base_dir = "micro-transformer"  # Change this if you want it elsewhere

# Create the main directory
os.makedirs(base_dir, exist_ok=True)

# Define the directory structure
directories = [
    "configs",
    "data",
    "tokenization",
    "model",
    "training",
    "distributed",
    "azure",
    "scripts",
    "notebooks"
]

# Create directories
for directory in directories:
    os.makedirs(os.path.join(base_dir, directory), exist_ok=True)

# Create Python package initialization files
for directory in ["data", "tokenization", "model", "training", "distributed"]:
    with open(os.path.join(base_dir, directory, "__init__.py"), "w") as f:
        f.write("# Package initialization\n")

# Create basic README.md
with open(os.path.join(base_dir, "README.md"), "w") as f:
    f.write("# Micro-Transformer Distributed Training\n\n")
    f.write("A project to implement a small transformer model with distributed training techniques.\n")

# Create .gitignore
with open(os.path.join(base_dir, ".gitignore"), "w") as f:
    f.write("# Python\n")
    f.write("__pycache__/\n*.py[cod]\n*$py.class\n.ipynb_checkpoints\n")
    f.write("\n# Virtual Environment\nvenv/\nenv/\n")
    f.write("\n# Model Checkpoints\ncheckpoints/\n*.pt\n*.bin\n")
    f.write("\n# Logs\nlogs/\n*.log\nruns/\n")
    f.write("\n# Data\ndata/raw/\ndata/processed/\n")

# Create requirements.txt
with open(os.path.join(base_dir, "requirements.txt"), "w") as f:
    f.write("torch>=2.0.0\n")
    f.write("transformers>=4.20.0\n")
    f.write("tokenizers>=0.13.0\n")
    f.write("tqdm>=4.64.0\n")
    f.write("wandb>=0.13.5\n")
    f.write("matplotlib>=3.5.3\n")
    f.write("numpy>=1.23.3\n")

# Create setup.py
with open(os.path.join(base_dir, "setup.py"), "w") as f:
    f.write("from setuptools import setup, find_packages\n\n")
    f.write("setup(\n")
    f.write("    name='micro-transformer',\n")
    f.write("    version='0.1.0',\n")
    f.write("    packages=find_packages(),\n")
    f.write("    install_requires=[\n")
    f.write("        'torch>=2.0.0',\n")
    f.write("        'transformers>=4.20.0',\n")
    f.write("        'tokenizers>=0.13.0',\n")
    f.write("        'tqdm>=4.64.0',\n")
    f.write("    ],\n")
    f.write(")\n")

# Create sample configuration files
model_config = {
    "vocab_size": 5000,
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 512,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512
}

with open(os.path.join(base_dir, "configs", "model_config.json"), "w") as f:
    json.dump(model_config, f, indent=4)

train_config = {
    "batch_size": 64,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "max_steps": 10000,
    "warmup_steps": 1000,
    "eval_steps": 500,
    "save_steps": 1000,
    "gradient_accumulation_steps": 1,
    "fp16": True
}

with open(os.path.join(base_dir, "configs", "train_config.json"), "w") as f:
    json.dump(train_config, f, indent=4)

distributed_config = {
    "data_parallel": {
        "world_size": 2,
        "backend": "nccl"
    },
    "pipeline_parallel": {
        "num_stages": 2,
        "num_microbatches": 4
    },
    "tensor_parallel": {
        "num_gpus": 2
    }
}

with open(os.path.join(base_dir, "configs", "distributed_config.json"), "w") as f:
    json.dump(distributed_config, f, indent=4)

# Create empty files for main modules
files_to_create = [
    # Data module
    ("data", "preprocessing.py"),
    ("data", "dataset.py"),
    ("data", "dataloader.py"),
    # Tokenization module
    ("tokenization", "tokenizer.py"),
    ("tokenization", "utils.py"),
    # Model module
    ("model", "layers.py"),
    ("model", "attention.py"),
    ("model", "transformer.py"),
    ("model", "utils.py"),
    # Training module
    ("training", "trainer.py"),
    ("training", "optimizer.py"),
    ("training", "checkpointing.py"),
    ("training", "metrics.py"),
    # Distributed module
    ("distributed", "data_parallel.py"),
    ("distributed", "pipeline_parallel.py"),
    ("distributed", "tensor_parallel.py"),
    ("distributed", "utils.py"),
    # Azure module
    ("azure", "vm_setup.sh"),
    ("azure", "distributed_setup.sh"),
    ("azure", "monitoring.py"),
    # Scripts
    ("scripts", "train.py"),
    ("scripts", "train_ddp.py"),
    ("scripts", "train_pipeline.py"),
    ("scripts", "train_tensor.py"),
    ("scripts", "benchmark.py"),
    ("scripts", "generate.py"),
    # Notebooks
    ("notebooks", "data_exploration.ipynb"),
    ("notebooks", "tokenizer_training.ipynb"),
    ("notebooks", "model_testing.ipynb")
]

for directory, filename in files_to_create:
    file_path = os.path.join(base_dir, directory, filename)
    with open(file_path, "w") as f:
        # Add a simple header to each file
        if filename.endswith(".py"):
            f.write(f"# {filename}\n# Implementation for the micro-transformer project\n\n")
        elif filename.endswith(".sh"):
            f.write("#!/bin/bash\n# Azure setup script\n\n")
        elif filename.endswith(".ipynb"):
            # Create a minimal valid notebook structure
            notebook_content = {
                "cells": [],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
            json.dump(notebook_content, f, indent=2)

print(f"Directory structure created at {os.path.abspath(base_dir)}")