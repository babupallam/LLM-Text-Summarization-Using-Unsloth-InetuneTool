import os
import pandas as pd
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu

# Define paths
processed_data_dir = "data/processed/"
model_save_dir = "models/"
results_dir = "results/"
os.makedirs(results_dir, exist_ok=True)  # Ensure results directory exists

# Load test data
test_data = pd.read_csv(os.path.join(processed_data_dir, "processed_test.csv"))

# Initialize evaluation metrics
rouge = load_metric("rouge")
results = []

# Define model names and paths
model_names = {
    "llama": "huggingface/llama-3.2",  # Replace with actual model identifier
    "mistral": "huggingface/mistral",  # Replace with actual model identifier
    "phi": "huggingface/phi-3.5",  # Replace with actual model identifier
    "gemma": "huggingface/gemma-2"  # Replace with actual model identifier
}


# Function to generate summary and calculate metrics
def evaluate_model(model_name, model_path, test_data):
    """
    Evaluates a given model on the test dataset using ROUGE and BLEU scores.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model directory.
        test_data (DataFrame): The test data containing articles and reference summaries.
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    rouge_scores = []
    bleu_scores = []

    print(f"Evaluating {model_name} model...")

    # Generate summaries and compute metrics
    for _, row in test_data.iterrows():
        inputs = tokenizer(row["article"], return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, length_penalty=2.0,
                                     early_stopping=True)
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Calculate ROUGE scores
        rouge_score = rouge.compute(predictions=[generated_summary], references=[row["summary"]])
        rouge_scores.append(rouge_score)

        # Calculate BLEU scores
        reference = row["summary"].split()
        candidate = generated_summary.split()
        bleu_score = sentence_bleu([reference], candidate)
        bleu_scores.append(bleu_score)

    # Average the metrics across all examples
    avg_rouge = {
        key: sum([score[key] for score in rouge_scores]) / len(rouge_scores) for key in rouge_scores[0]
    }
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    # Append results
    results.append({
        "model": model_name,
        "average_rouge": avg_rouge,
        "average_bleu": avg_bleu
    })

    print(f"{model_name} evaluation complete.")
    print(f"Average ROUGE: {avg_rouge}")
    print(f"Average BLEU: {avg_bleu}")


# Evaluate each model
for model_name, model_path in model_names.items():
    evaluate_model(model_name, os.path.join(model_save_dir, model_name), test_data)

# Save results to a CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(results_dir, "evaluation_results.csv"), index=False)
print("Evaluation results saved to results/evaluation_results.csv")
