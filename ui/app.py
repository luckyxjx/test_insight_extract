from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Initialize the BERT Summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        text = request.json.get('text', '')
        if not text.strip():
            return jsonify({"error": "Text input is empty!"}), 400

        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
