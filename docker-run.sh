#!/bin/bash
# Docker wrapper for NVFP4 quantization on DGX Spark (aarch64)
# This script runs the quantization workflow inside NVIDIA PyTorch container

set -e

CONTAINER_IMAGE="nvcr.io/nvidia/pytorch:25.10-py3"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

echo "Starting NVFP4 Quantization in Docker Container"
echo "Container: $CONTAINER_IMAGE"
echo "Project: $PROJECT_DIR"
echo "HF Cache: $HF_CACHE_DIR"
echo ""

# Install required packages in container and run script
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
        # Set CUDA library paths for vLLM compatibility
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        export CUDA_HOME=/usr/local/cuda

        # Install required packages with compatible versions for nvidia-modelopt
        echo "[Container] Installing required packages..."

        # Check if we are running inference script that needs vLLM
        if echo "$@" | grep -q "inference_vllm"; then
            echo "[Container] Setting up CUDA 12 compatibility for vLLM..."
            # vLLM expects CUDA 12.x but container has CUDA 13.0 - create symlink
            (cd /usr/local/cuda/lib64 && ln -sf libcudart.so.13 libcudart.so.12)

            echo "[Container] Installing vLLM for NVFP4 inference..."
            pip install --quiet vllm>=0.6.5 nvidia-modelopt[hf] datasets>=3.1.0 huggingface-hub>=0.26.0 safetensors>=0.4.5
        else
            pip install --quiet nvidia-modelopt[hf] datasets>=3.1.0 huggingface-hub>=0.26.0 safetensors>=0.4.5
        fi

        # Run the script passed as argument
        echo "[Container] Running: $@"
        python "$@"
    ' -- "$@"
