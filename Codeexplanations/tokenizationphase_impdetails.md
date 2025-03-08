# Micro-Transformer Tokenization Implementation: Step-by-Step Guide

To complete your understanding of the tokenization phase, here's a step-by-step guide for using the tokenizer in your micro-transformer project:

## Step 1: Training the Tokenizer

Train a new tokenizer on your dataset using the training script:

```bash
python scripts/train_tokenizer.py --data_dir data/ --output_dir micro_tokenization/pretrained --vocab_size 5000
```

This will:
1. Find all training files in the data directory
2. Create a temporary file with all the text
3. Train a ByteLevelBPE tokenizer with 5000 tokens
4. Save the tokenizer files to the specified output directory
5. Test the tokenizer on sample sentences

## Step 2: Testing the Tokenizer

Verify the tokenizer works correctly with the testing script:

```bash
python scripts/test_tokenizer.py --tokenizer_path micro_tokenization/pretrained
```

This will run a series of tests on your tokenizer, including:
- Basic encoding and decoding
- Truncation and padding
- Batch processing
- Causal and masked language modeling input creation

## Step 3: Using the Tokenizer in Your Model

Here's how to use the tokenizer in your model training pipeline:

```python
from micro_tokenization import MicroTransformerTokenizer, prepare_input_for_batch

# Load the trained tokenizer
tokenizer = MicroTransformerTokenizer.from_pretrained("micro_tokenization/pretrained")

# For simple language modeling with your transformer
def prepare_training_batch(texts, max_length=128):
    inputs = prepare_input_for_batch(
        texts,
        tokenizer,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return inputs

# In your training loop
batch_texts = ["Example sentence 1", "Example sentence 2"]
inputs = prepare_training_batch(batch_texts)
outputs = model(**inputs)  # Forward pass through your model
```

## Step 4: Integrating with Model Architecture

When you implement your transformer model in the next phase, this is how the tokenizer will connect:

```python
class MicroTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The embedding layer will use the vocabulary size from the tokenizer
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # Rest of the transformer architecture...
    
    def forward(self, input_ids, attention_mask=None):
        # input_ids comes from the tokenizer
        embeddings = self.embeddings(input_ids)
        # Rest of the forward pass...
        return outputs
```

## Step 5: Inference with the Tokenizer

During inference (text generation), you'll use the tokenizer like this:

```python
def generate_text(model, prompt, max_length=50):
    # Encode the prompt
    inputs = tokenizer.encode(
        prompt, 
        return_tensors="pt",
        add_special_tokens=True
    )
    
    # Generate tokens
    input_ids = inputs["input_ids"]
    for _ in range(max_length):
        # Forward pass
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        
        # Get the next token (greedy decoding)
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Append to the sequence
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Stop if we generate an EOS token
        if next_token.item() == tokenizer.token_to_id("[SEP]"):
            break
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    return generated_text
```

This completes the tokenization phase of your project and sets you up for the model architecture design phase.