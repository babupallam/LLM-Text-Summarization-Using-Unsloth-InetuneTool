import os
import pandas as pd
from datasets import load_dataset

# Define paths
raw_data_dir = "../data/raw/"
os.makedirs(raw_data_dir, exist_ok=True)  # Ensure the raw data directory exists

# Load the CNN/Daily Mail dataset
print("Loading CNN/Daily Mail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0")


# Function to save each split to a CSV file
def save_split_to_csv(split_name, split_data):
    """
    Saves a dataset split to a CSV file in the raw data directory.

    Args:
        split_name (str): The name of the dataset split (e.g., 'train', 'validation', 'test').
        split_data (Dataset): The dataset split to save.
    """
    # Convert to pandas DataFrame
    df = pd.DataFrame(split_data)

    # Define the output file path
    output_file_path = os.path.join(raw_data_dir, f"cnn_dailymail_{split_name}.csv")

    # Save the DataFrame to CSV
    df.to_csv(output_file_path, index=False)
    print(f"{split_name.capitalize()} split saved to {output_file_path}")


# Save each dataset split
for split_name in ["train", "validation", "test"]:
    save_split_to_csv(split_name, dataset[split_name])

print("All splits saved successfully!")
