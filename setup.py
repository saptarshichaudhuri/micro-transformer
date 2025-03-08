from setuptools import setup, find_packages

setup(
    name="micro-transformer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.20.0",
        "tokenizers>=0.13.0",
        "tqdm>=4.64.0",
        "numpy>=1.22.0",
    ],
    author="Saptarshi Chaudhuri",
    author_email="your.email@example.com",
    description="A micro-transformer implementation for distributed training",
)