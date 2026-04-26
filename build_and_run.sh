#!/bin/bash
# ============================================================
# TruthLens — Build & Run Script
# SEAI Project | SDG 16 — Peace, Justice & Strong Institutions
# ============================================================

set -e

IMAGE_NAME="truthlens-fake-news-detector"
CONTAINER_NAME="truthlens"
PORT=5000

echo "=================================================="
echo "  TruthLens — AI Fake News Detector"
echo "  SEAI Project | SDG 16"
echo "=================================================="

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed. Please install Docker first."
    exit 1
fi

# Stop and remove existing container if running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "[INFO] Stopping existing container..."
    docker stop $CONTAINER_NAME
fi

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "[INFO] Removing existing container..."
    docker rm $CONTAINER_NAME
fi

# Build the Docker image
echo "[INFO] Building Docker image: $IMAGE_NAME ..."
docker build -t $IMAGE_NAME .

echo "[INFO] Build complete!"

# Run the container
echo "[INFO] Starting container on port $PORT ..."
docker run -d \
  --name $CONTAINER_NAME \
  -p $PORT:5000 \
  -e GROQ_API_KEY="${GROQ_API_KEY}" \
  -e BERT_MODEL="${BERT_MODEL:-mrm8488/bert-tiny-finetuned-fake-news-detection}" \
  -e GROQ_MODEL="${GROQ_MODEL:-llama3-8b-8192}" \
  $IMAGE_NAME

echo ""
echo "[SUCCESS] TruthLens is running!"
echo "  Local URL : http://localhost:$PORT"
echo "  Health    : http://localhost:$PORT/health"
echo "  API       : POST http://localhost:$PORT/predict"
echo ""
echo "  To set your Groq API key:"
echo "  export GROQ_API_KEY=your_key_here && bash build_and_run.sh"
echo ""
echo "  To view logs:"
echo "  docker logs -f $CONTAINER_NAME"
echo "=================================================="
