from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load model directly from Hugging Face Hub
classifier = pipeline(
    "emotion-classification",
    model="shaadclt/distilbert-emotion-classifier"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text")
    result = classifier(text)[0]
    return jsonify({
        "emotion": result["label"],
        "confidence": round(result["score"] * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
