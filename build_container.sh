#!/bin/bash
set -e

# Build the container
echo "Building CaptchaKraken Container..."
# Pass HF_TOKEN if available to bake models in (optional for public models)
if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN not set. Assuming public models."
    docker build -t captchakraken-vllm-5090 .
else
    docker build --build-arg HF_TOKEN=$HF_TOKEN -t captchakraken-vllm-5090 .
fi

echo "Build complete."
echo "To run:"
echo "docker run --gpus all -p 8000:8000 --ipc=host -e VLLM_API_KEY=your_key captchakraken-vllm-5090"
