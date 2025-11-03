#!/bin/bash
# Docker wrapper for NVFP4 inference testing using official NVIDIA vLLM container
# This script runs inference on quantized models using nvcr.io/nvidia/vllm:25.10-py3

set -e

CONTAINER_IMAGE="nvcr.io/nvidia/vllm:25.10-py3"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

echo "Starting vLLM NVFP4 Inference in Docker Container"
echo "Container: $CONTAINER_IMAGE"
echo "Project: $PROJECT_DIR"
echo "HF Cache: $HF_CACHE_DIR"
echo ""

# Run inference script in vLLM container
docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$PROJECT_DIR:/workspace" \
    -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
    -w /workspace \
    "$CONTAINER_IMAGE" \
    bash -c '
        # Install additional dependencies needed for our scripts
        echo "[Container] Installing additional dependencies..."
        pip install --quiet nvidia-modelopt[hf] datasets>=3.1.0 huggingface-hub>=0.26.0 safetensors>=0.4.5

        # Run the script passed as argument
        echo "[Container] Running: $@"
        python "$@"
    ' -- "$@"
