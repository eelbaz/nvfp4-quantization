#!/bin/bash
#
# Start vLLM Server for NVFP4 Quantized Model
# ===========================================
#
# This script starts a vLLM server with OpenAI-compatible API for the NVFP4 quantized model.
# The server will be accessible to Open WebUI at http://localhost:8355/v1
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
PROJECT_DIR="$SCRIPT_DIR"
QUANTIZED_MODEL_DIR="$PROJECT_DIR/quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
CONTAINER_NAME="vllm-nvfp4-server"
HOST_PORT="8355"
CONTAINER_PORT="8000"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "vLLM NVFP4 Model Server Startup"
echo "========================================"
echo ""

# Check if quantized model exists
if [ ! -d "$QUANTIZED_MODEL_DIR" ]; then
    echo -e "${RED}Error: Quantized model not found at $QUANTIZED_MODEL_DIR${NC}"
    echo "Please run quantization first: ./docker-run.sh scripts/02_quantize_to_nvfp4.py"
    exit 1
fi

echo -e "${GREEN}Found quantized model at: $QUANTIZED_MODEL_DIR${NC}"

# Check if container already running
if docker ps | grep -q "$CONTAINER_NAME"; then
    echo -e "${YELLOW}Container $CONTAINER_NAME is already running.${NC}"
    echo "Stop it with: docker stop $CONTAINER_NAME"
    echo "Or remove it with: docker rm -f $CONTAINER_NAME"
    exit 1
fi

# Remove stopped container if exists
if docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo -e "${YELLOW}Removing stopped container...${NC}"
    docker rm "$CONTAINER_NAME"
fi

echo ""
echo "Starting vLLM server..."
echo "  Model: Qwen3-VLTO-32B-Instruct-NVFP4"
echo "  Port: $HOST_PORT (mapped to container port $CONTAINER_PORT)"
echo "  API: http://localhost:$HOST_PORT/v1"
echo "  Container: $CONTAINER_NAME"
echo ""

# Start vLLM server with OpenAI-compatible API
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "$HOST_PORT:$CONTAINER_PORT" \
    -v "$QUANTIZED_MODEL_DIR:/models/quantized:ro" \
    -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    nvcr.io/nvidia/vllm:25.10-py3 \
    vllm serve /models/quantized \
        --quantization modelopt \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --max-model-len 32768 \
        --served-model-name "Qwen3-VLTO-32B-Instruct-NVFP4" \
        --host 0.0.0.0 \
        --port $CONTAINER_PORT

echo ""
echo -e "${GREEN}Waiting for vLLM server to start...${NC}"
sleep 10

# Check if container is running
if docker ps | grep -q "$CONTAINER_NAME"; then
    echo -e "${GREEN}vLLM server started successfully!${NC}"
    echo ""
    echo "Server Information:"
    echo "  API Endpoint: http://localhost:$HOST_PORT/v1"
    echo "  Models Endpoint: http://localhost:$HOST_PORT/v1/models"
    echo "  Health Check: http://localhost:$HOST_PORT/health"
    echo ""
    echo "Open WebUI Configuration:"
    echo "  Your Open WebUI is already configured to use: http://localhost:8355/v1"
    echo "  The model should appear automatically in Open WebUI"
    echo ""
    echo "Useful Commands:"
    echo "  View logs: docker logs -f $CONTAINER_NAME"
    echo "  Stop server: docker stop $CONTAINER_NAME"
    echo "  Restart server: docker restart $CONTAINER_NAME"
    echo ""
    echo "Test the API with:"
    echo '  curl http://localhost:8355/v1/models'
    echo ""
else
    echo -e "${RED}Error: Container failed to start${NC}"
    echo "Check logs with: docker logs $CONTAINER_NAME"
    exit 1
fi
