#!/bin/bash
#
# Stop vLLM Server
# ================
#

CONTAINER_NAME="vllm-nvfp4-server"

echo "Stopping vLLM server..."

if docker ps | grep -q "$CONTAINER_NAME"; then
    docker stop "$CONTAINER_NAME"
    echo "Server stopped."
else
    echo "Server is not running."
fi

# Optionally remove container
read -p "Remove container? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker rm "$CONTAINER_NAME" 2>/dev/null && echo "Container removed."
fi
