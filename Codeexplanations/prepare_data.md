python scripts/prepare_data.py --max_samples 50000

I'll walk you through what both the `preprocessing.py` and `prepare_data.py` scripts do in detail.

### `preprocessing.py` Explanation

This script provides the core functionality for processing the TinyStories dataset:

1. **`load_tinystories()` function**:
   - Uses Hugging Face's `datasets` library to download and load the TinyStories dataset
   - This approach handles caching properly, so the dataset is only downloaded once
   - The function returns the dataset object with train and validation splits

2. **`preprocess_text(raw_text)` function**:
   - Takes a raw text string and cleans it:
     - Removes excessive newlines and creates a properly spaced single paragraph
     - Replaces double spaces with single spaces
   - This standardizes the text format for more consistent training

3. **`extract_samples_from_dataset(dataset, split, max_samples, min_length, max_length)` function**:
   - Takes a dataset object and extracts examples from a specific split (train/validation)
   - Applies filters based on text length to ensure suitable examples
   - Process:
     - Attempts to access the text field in each example (trying 'text' then 'story')
     - Calls `preprocess_text()` to clean each story
     - Keeps only stories with length between `min_length` and `max_length`
     - Stops after processing `max_samples` examples (if specified)
   - Returns a list of cleaned text samples

4. **`create_dataset_splits(max_samples, min_length, max_length)` function**:
   - Coordinates the overall dataset creation process:
     - Loads the TinyStories dataset
     - Extracts samples from the train split
     - Extracts samples from validation split (or creates one from train if needed)
     - Creates a test set from validation (or train if validation is too small)
   - Returns a dictionary with 'train', 'validation', and 'test' splits

5. **`save_splits(splits, output_dir)` function**:
   - Takes the dataset splits and saves them to disk in two formats:
     - Text files (`.txt`) with stories separated by double newlines
     - JSON Lines files (`.jsonl`) with each line containing a JSON object with a "text" field
   - Creates the output directory if it doesn't exist

6. **`main()` function**:
   - Parses command-line arguments for customizing the data processing
   - Calls `create_dataset_splits()` to generate the dataset
   - Saves the full dataset splits
   - Creates and saves a smaller subset for quick testing
   - Prints summary information about the dataset

### `prepare_data.py` Explanation

This script is a simpler wrapper around the functionality in `preprocessing.py`:

1. **Imports**:
   - Adds the project root to the Python path so it can import from the `data` package
   - Imports the needed functions from `preprocessing`

2. **`main()` function**:
   - Parses command-line arguments:
     - `--processed_dir`: Where to save the processed data
     - `--max_samples`: Maximum number of samples to extract
     - `--min_length` and `--max_length`: Text length filters
   - Creates the output directory
   - Calls `create_dataset_splits()` to generate the dataset
   - Saves the full dataset
   - Creates and saves a smaller subset for testing (1000 train samples, 100 each for validation/test)
   - Prints completion messages with dataset locations

### What Happened During Execution

When you ran the script with `--max_samples 50000`:

1. The script downloaded the TinyStories dataset (~1GB) from Hugging Face
2. It processed the first 50,000 examples from the train split, but only kept 1,803 that met the length criteria
3. It processed 5,000 examples from the validation split and kept 196 that met the criteria
4. It then split those 196 validation samples into 98 validation and 98 test samples
5. It saved the data in both text and JSON formats
6. It created a smaller dataset with 1,000 training samples for quick experiments

The length filtering (150-500 characters) seems quite strict, which is why you ended up with fewer samples than requested. This is actually beneficial for your project as it keeps the dataset manageable while still providing enough examples for training a small transformer.

### Key Features of the Implementation

1. **Efficiency**: Uses Hugging Face's optimized data loading
2. **Flexibility**: Command-line arguments for customization
3. **Multiple formats**: Saves in both text and JSON formats
4. **Test subset**: Creates a smaller dataset for quick iteration
5. **Quality control**: Filters samples by length to ensure consistent quality
6. **Progress tracking**: Uses tqdm for progress bars during processing

This data preparation approach provides a solid foundation for the next step in your project: tokenizer development and training.