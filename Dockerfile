# ==============================
# TruthLens — Fake News Detector
# SEAI Project | SDG 16
# ==============================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the BERT model during build (so container starts fast)
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='mrm8488/bert-tiny-finetuned-fake-news-detection')"

# Copy application code
COPY app/ .

# Environment variables (override at runtime)
ENV GROQ_API_KEY=""
ENV BERT_MODEL="mrm8488/bert-tiny-finetuned-fake-news-detection"
ENV GROQ_MODEL="llama3-8b-8192"
ENV PORT=5000

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
