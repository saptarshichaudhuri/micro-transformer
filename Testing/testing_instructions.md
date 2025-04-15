# MicroTransformer Testing Guide

This guide explains how to test and validate your trained MicroTransformer model using the `model_test.py` script.

## Testing Capabilities

The testing script provides several capabilities for evaluating your trained model:

1. **Text Completion**: Generate continuations for input prompts
2. **Perplexity Evaluation**: Calculate perplexity scores on test inputs
3. **Sample Generation Analysis**: Analyze diversity and quality of generated samples

## Usage

```bash
python model_test.py \
  --model_path /path/to/model/checkpoint.pt \
  --tokenizer_path /path/to/tokenizer/directory \
  --test_file /path/to/prompts.txt \  # Optional: provide your own test prompts
  --max_length 100 \
  --temperature 0.8 \
  --top_k 50 \
  --top_p 0.9 \
  --num_samples 3 \
  --output_dir test_results \
  --device cuda  # or cpu
```


### Arguments

- `--model_path`: Path to the saved model checkpoint (.pt file)
- `--tokenizer_path`: Path to the tokenizer directory (containing vocab.json and merges.txt)
- `--test_file`: (Optional) Path to a file containing test prompts (one per line)
- `--max_length`: Maximum number of tokens to generate
- `--temperature`: Sampling temperature (higher = more random, lower = more deterministic)
- `--top_k`: Top-k sampling parameter (number of highest probability tokens to consider)
- `--top_p`: Top-p (nucleus) sampling parameter (cumulative probability threshold)
- `--num_samples`: Number of samples to generate per prompt
- `--output_dir`: Directory to save test results
- `--device`: Device to run on (cuda/cpu)

## Test Results

The script generates the following outputs in the specified output directory:

1. **JSON Files**:
   - `completion_results.json`: Text completion results for each prompt and sample
   - `perplexity_results.json`: Perplexity scores for test prompts
   - `diversity_results.json`: Analysis of generation diversity and quality
   - `test_config.json`: Configuration used for the test

2. **Visualizations**:
   - `perplexity_distribution.png`: Distribution of perplexity scores
   - `token_diversity_distribution.png`: Distribution of token diversity scores
   - `sample_length_distribution.png`: Distribution of sample lengths
   - `length_vs_diversity.png`: Relationship between sample length and diversity

## Examples

### Basic Testing

```bash
python test_model.py \
  --model_path checkpoints/checkpoint-best.pt \
  --tokenizer_path micro_tokenization/pretrained
```

### Custom Testing with More Randomness

```bash
python test_model.py \
  --model_path checkpoints/checkpoint-best.pt \
  --tokenizer_path micro_tokenization/pretrained \
  --test_file my_test_prompts.txt \
  --temperature 1.2 \
  --num_samples 5
```

### Evaluating Deterministic Outputs

```bash
python test_model.py \
  --model_path checkpoints/checkpoint-best.pt \
  --tokenizer_path micro_tokenization/pretrained \
  --temperature 0.2 \
  --top_k 10 \
  --top_p 0.5
```
```

This comprehensive implementation provides:

1. **Text Completion Testing**: The script generates continuations for input prompts, allowing you to evaluate how well the model completes text.

2. **Perplexity Evaluation**: It calculates perplexity scores for test inputs, which is a standard metric for language models.

3. **Sample Generation Analysis**: The script analyzes the diversity and quality of generated samples, including token diversity metrics and visualizations.

4. **Visualizations**: It creates several plots to visualize the model's performance, including perplexity distribution, token diversity, and sample length.

5. **Result Saving**: All results are saved to JSON files and visualizations for further analysis.

The script is designed to work with your existing MicroTransformer architecture and components, leveraging the functionality you've already implemented. You can easily extend it to include additional metrics or analyses as needed.