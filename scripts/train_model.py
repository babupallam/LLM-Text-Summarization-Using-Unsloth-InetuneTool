import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset

# Define paths
processed_data_dir = "data/processed/"
model_save_dir = "models/"
os.makedirs(model_save_dir, exist_ok=True)

# Choose model names for each experiment
model_names = {
    "llama": "huggingface/llama-3.2",   # Replace with actual model identifier
    "mistral": "huggingface/mistral",   # Replace with actual model identifier
    "phi": "huggingface/phi-3.5",       # Replace with actual model identifier
    "gemma": "huggingface/gemma-2"      # Replace with actual model identifier
}

# Load processed data
def load_data(split_name):
    """
    Loads processed data from CSV and converts it to Hugging Face Dataset format.
    """
    file_path = os.path.join(processed_data_dir, f"processed_{split_name}.csv")
    df = pd.read_csv(file_path)
    return Dataset.from_pandas(df)

# Prepare dataset for training
def preprocess_data(examples, tokenizer, max_input_length=512, max_output_length=128):
    """
    Tokenizes the article and summary columns.
    """
    inputs = tokenizer(examples["article"], max_length=max_input_length, truncation=True, padding="max_length")
    targets = tokenizer(examples["summary"], max_length=max_output_length, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

# Training function
def train_model(model_name, model_path, tokenizer, train_data, val_data):
    """
    Trains the model with the specified tokenizer and dataset.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenized_train_data = train_data.map(lambda x: preprocess_data(x, tokenizer), batched=True)
    tokenized_val_data = val_data.map(lambda x: preprocess_data(x, tokenizer), batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(model_save_dir, model_name),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_val_data,
        tokenizer=tokenizer
    )

    # Train the model
    print(f"Training {model_name} model...")
    trainer.train()
    model.save_pretrained(os.path.join(model_save_dir, model_name))
    tokenizer.save_pretrained(os.path.join(model_save_dir, model_name))
    print(f"Model {model_name} saved successfully!")

# Main training loop
def main():
    # Load data splits
    train_data = load_data("train")
    val_data = load_data("validation")

    # Train each model
    for model_name, model_path in model_names.items():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        train_model(model_name, model_path, tokenizer, train_data, val_data)

if __name__ == "__main__":
    main()
# Script for training the model