"""
Usage: in your notebooks, you can simply import and use these functions:

1. Open a new Colab notebook
2. Run a cell to clone your GitHub repository:

!git clone https://github.com/yourusername/micro-transformer.git
%cd micro-transformer

3. Then import and run your setup script:
from scripts.colab_setup import check_gpu, install_packages, sync_repo

# Check GPU
check_gpu()

# Install packages
install_packages()

# Mount Drive
# You don't necessarily need to mount Google Drive unless you want to persist data between Colab sessions.
mount_drive('/content/drive/MyDrive/micro-transformer')

# Sync with repo
sync_repo('https://github.com/yourusername/micro-transformer.git', '/content/micro-transformer')

"""




# scripts/colab_setup.py

def check_gpu():
    """Verify GPU availability and print information"""
    import torch
    from subprocess import getoutput
    
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        gpu_info = getoutput('nvidia-smi')
        print("GPU Info:")
        print(gpu_info)
        print("\nCUDA Device:", torch.cuda.get_device_name(0))
        print("Number of GPUs:", torch.cuda.device_count())
        return True
    else:
        print("WARNING: No GPU detected. Go to Runtime > Change runtime type and select GPU.")
        return False

def mount_drive(project_path):
    """Mount Google Drive and navigate to project directory"""
    from google.colab import drive
    import os
    
    drive.mount('/content/drive')
    
    os.chdir(project_path)
    print(f"Working directory: {os.getcwd()}")
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

def install_packages():
    """Install required packages"""
    import sys
    import subprocess
    
    packages = [
        "torch>=2.0.0",
        "transformers>=4.20.0",
        "tokenizers>=0.13.0",
        "tqdm>=4.64.0",
        "wandb>=0.13.5",
        "matplotlib>=3.5.3",
        "numpy>=1.23.3"
    ]
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
    
    # Verify installations
    import torch
    import transformers
    import tokenizers
    import tqdm
    import wandb
    import matplotlib
    import numpy
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Tokenizers version: {tokenizers.__version__}")
    print(f"Tqdm version: {tqdm.__version__}")
    print(f"Wandb version: {wandb.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Numpy version: {numpy.__version__}")

def sync_repo(repo_url, repo_dir):
    """Sync with GitHub repository"""
    import os
    import sys
    import subprocess
    
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url, repo_dir])
        print(f"Repository cloned to {repo_dir}")
    else:
        print(f"Repository already exists at {repo_dir}")
    
    # Change to the repository directory
    os.chdir(repo_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Pull the latest changes
    subprocess.run(["git", "pull"])
    print("Repository updated to the latest version")
    
    # Add Python path for imports
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
        print(f"Added {repo_dir} to Python path")