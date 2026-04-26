"""
SEAI Project: Fake News / Misinformation Detector + Fact-Check Explainer
SDG 16 - Peace, Justice & Strong Institutions
Inference Engine: Flask REST API
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Import modules after app creation to avoid circular issues
from model import classify_news
from llm import explain_with_llm


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": "BERT Fake News Detector + LLaMA 3 Explainer",
        "sdg": "SDG 16 - Peace, Justice & Strong Institutions",
        "version": "1.0.0"
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided. Send JSON with 'text' field."}), 400

        text = data['text'].strip()
        if len(text) < 15:
            return jsonify({"error": "Text too short. Please provide at least 15 characters."}), 400

        logger.info(f"Classifying text of length {len(text)}")

        # Step 1: BERT Classification
        classification = classify_news(text)
        logger.info(f"Classification result: {classification}")

        # Step 2: LLM Explanation via Groq
        explanation = explain_with_llm(text, classification)
        logger.info("LLM explanation generated successfully")

        return jsonify({
            "verdict": classification['label'],
            "confidence": classification['score'],
            "explanation": explanation['analysis'],
            "red_flags": explanation['red_flags'],
            "credibility_score": explanation['credibility_score'],
            "fact_check_summary": explanation['fact_check_summary'],
            "text_preview": text[:200] + "..." if len(text) > 200 else text
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
