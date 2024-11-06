from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize Flask app
app = Flask(__name__)

# Load the best model and tokenizer (update model path as needed)
model_name = "models/llama"  # Change to your best-performing model directory
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define the home route
@app.route("/")
def index():
    return render_template("index.html")

# Define the summarize route
@app.route("/summarize", methods=["POST"])
def summarize():
    # Get the input text from the form
    input_text = request.form["input_text"]

    # Tokenize and generate the summary
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Render the result on the page
    return render_template("index.html", input_text=input_text, summary=summary)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
# Main backend file to handle web requests