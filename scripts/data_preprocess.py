import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Ensure NLTK stopwords are available
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Define paths
raw_data_dir = "data/raw/"
processed_data_dir = "data/processed/"
os.makedirs(processed_data_dir, exist_ok=True)  # Ensure the processed data directory exists

# Load English stopwords
stop_words = set(stopwords.words("english"))

# Enhanced text cleaning function
def clean_text(text):
    """
    Cleans the input text by applying various techniques:
    - Lowercasing
    - Removing special characters and numbers
    - Removing stop words
    - Removing extra whitespace and punctuation

    Args:
        text (str): The raw text data.

    Returns:
        str: The cleaned text.
    """
    # 1. Lowercasing
    text = text.lower()

    # 2. Remove URLs and HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)

    # 3. Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", '', text)

    # 4. Tokenize and remove stop words
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]

    # 6. Remove short words (optional, to remove words less than 3 characters)
    tokens = [word for word in tokens if len(word) > 2]

    # 7. Join tokens back into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text

# Function to preprocess a dataset split and save it
def preprocess_and_save_split(split_name):
    """
    Loads, cleans, and saves the preprocessed data for a specific dataset split.

    Args:
        split_name (str): The name of the dataset split (e.g., 'train', 'validation', 'test').
    """
    # Define input and output file paths
    input_file_path = os.path.join(raw_data_dir, f"cnn_dailymail_{split_name}.csv")
    output_file_path = os.path.join(processed_data_dir, f"processed_{split_name}.csv")

    # Load the raw CSV file
    print(f"Loading {split_name} split...")
    df = pd.read_csv(input_file_path)
    print("Columns in dataset:", df.columns)  # Debug: Print column names to verify

    # Clean the article and summary columns if they exist
    print(f"Preprocessing {split_name} split...")
    if 'article' in df.columns:
        df['article'] = df['article'].apply(clean_text)
    else:
        print(f"Warning: 'article' column not found in {split_name} split.")

    if 'summary' in df.columns:
        df['summary'] = df['summary'].apply(clean_text)
    else:
        print(f"Warning: 'summary' column not found in {split_name} split.")

    # Save the cleaned data to CSV
    df.to_csv(output_file_path, index=False)
    print(f"Processed {split_name} split saved to {output_file_path}")

# Preprocess and save each dataset split
for split in ["train", "validation", "test"]:
    preprocess_and_save_split(split)

print("All splits processed and saved successfully!")
