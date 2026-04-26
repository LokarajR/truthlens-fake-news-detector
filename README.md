# TruthLens 🔍 — AI-Powered Fake News Detector

[![SDG 16](https://img.shields.io/badge/SDG-16%20Peace%20%26%20Justice-blue)](https://sdgs.un.org/goals/goal16)
[![Docker](https://img.shields.io/badge/Docker-Hub-blue)](https://hub.docker.com)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-lightgrey)](https://flask.palletsprojects.com)

> SEAI Individual Project | SDG 16 — Peace, Justice & Strong Institutions

## 🌐 What It Does

**TruthLens** detects misinformation in news articles using a two-stage AI pipeline:

1. **BERT NLP Model** — Classifies the article as REAL or FAKE
2. **LLaMA 3 via Groq** — Explains *why*, flags suspicious patterns, and gives a credibility score

## 🏗️ Architecture

```
User Input (News text)
       ↓
Flask REST API (/predict)
       ↓
BERT Classifier (mrm8488/bert-tiny-finetuned-fake-news-detection)
       ↓
Groq LLM (llama3-8b-8192) — Fact-Check Explanation
       ↓
JSON Response → Beautiful Web UI
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/truthlens-fake-news-detector
cd truthlens-fake-news-detector

# Set your Groq API key
export GROQ_API_KEY=your_groq_api_key_here

# Build and run
bash build_and_run.sh
```

Open: http://localhost:5000

### Option 2: Run Locally

```bash
cd app
pip install -r requirements.txt
export GROQ_API_KEY=your_groq_api_key_here
python app.py
```

### Option 3: Pull from DockerHub

```bash
docker pull YOUR_DOCKERHUB_USERNAME/truthlens-fake-news-detector:latest
docker run -p 5000:5000 -e GROQ_API_KEY=your_key YOUR_DOCKERHUB_USERNAME/truthlens-fake-news-detector
```

## 📡 API Reference

### POST `/predict`
```json
Request:  { "text": "Your news article here..." }
Response: {
  "verdict": "FAKE",
  "confidence": 94.2,
  "explanation": "This article uses...",
  "red_flags": ["Sensationalist language", "..."],
  "credibility_score": 12,
  "fact_check_summary": "AI classifies this as likely FAKE news."
}
```

### GET `/health`
```json
{ "status": "healthy", "model": "BERT Fake News Detector + LLaMA 3 Explainer" }
```

## 🧪 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required for LLM)* | Get free at console.groq.com |
| `BERT_MODEL` | `mrm8488/bert-tiny-finetuned-fake-news-detection` | HuggingFace model ID |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq LLM model name |
| `PORT` | `5000` | Flask server port |

## 📁 Project Structure

```
seai_proj_/
├── app/
│   ├── app.py              # Flask server (inference engine)
│   ├── model.py            # BERT classifier
│   ├── llm.py              # Groq LLM integration
│   ├── requirements.txt
│   └── templates/
│       └── index.html      # Web UI
├── Dockerfile
├── build_and_run.sh        # Shell script (containerize)
├── .gitignore
└── README.md
```

## 🎓 SEAI Project Requirements

| Requirement | Implementation |
|-------------|----------------|
| ML/DL Application | BERT fine-tuned fake news classifier |
| LLM/GenAI Integration | Groq API with LLaMA 3 |
| Inference Engine | Flask REST API |
| Containerization | Docker + `build_and_run.sh` |
| GitHub | This repository |
| DockerHub | `YOUR_USERNAME/truthlens-fake-news-detector` |
| SDG Alignment | SDG 16 — Peace, Justice & Strong Institutions |

## 📄 License

MIT License — Open source for research and educational use.
