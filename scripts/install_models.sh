#!/bin/bash

# scripts/install_models.sh
# Installs Ollama and sets up Holo2 and Qwen3-VL models based on system RAM.

set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "=== CaptchaKraken Model Installer ==="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check Hardware Requirements
if command_exists python3; then
    echo "[*] Checking hardware requirements..."
    if ! python3 "$SCRIPT_DIR/check_hw.py"; then
        echo ""
        echo "Error: Hardware requirements not met for local models."
        echo "Please configure the application to use an external API (e.g., OpenAI)."
        exit 1
    fi
else
    echo "Warning: Python3 not found. Skipping hardware check."
fi

# 1. Install Ollama if not present
if ! command_exists ollama; then
    echo "[*] Ollama not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command_exists brew; then
            echo "[*] macOS detected. Installing Ollama via Homebrew..."
            brew install ollama
        else
            echo "Error: Homebrew not found. Please install Homebrew or install Ollama manually."
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux"* ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "Error: Unsupported OS for automatic installation."
        echo "Please install Ollama manually from https://ollama.com"
        exit 1
    fi
else
    echo "[*] Ollama is already installed."
fi

# 2. Ensure Ollama Server is running
if ! pgrep -x "ollama" > /dev/null && ! pgrep -f "ollama serve" > /dev/null; then
    echo "[*] Starting Ollama server..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    # Wait for server to start
    echo "    Waiting for Ollama to initialize..."
    sleep 5
else
    echo "[*] Ollama server is running."
fi

# 3. Detect RAM
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    RAM_BYTES=$(sysctl -n hw.memsize)
elif [ "$OS" = "Linux" ]; then
    RAM_BYTES=$(grep MemTotal /proc/meminfo | awk '{print $2 * 1024}')
else
    echo "Warning: Cannot detect RAM. Defaulting to 8GB assumption."
    RAM_BYTES=8589934592
fi

RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))
echo "[*] Detected System RAM: ${RAM_GB} GB"

# 4. Select Models
# Logic:
# < 16GB:  Holo2 4B, Qwen3-VL 2B/4B
# 16-32GB: Holo2 8B, Qwen3-VL 8B
# > 32GB:  Holo2 30B, Qwen3-VL 32B (or largest available)

HOLO_MODEL=""
QWEN_MODEL=""

if [ "$RAM_GB" -lt 16 ]; then
    echo "    -> Low RAM detected (<16GB). Selecting lightweight models."
    HOLO_MODEL="holo2:4b"
    QWEN_MODEL="qwen3-vl:2b" # 2B is very safe for low RAM
elif [ "$RAM_GB" -lt 32 ]; then
    echo "    -> Medium RAM detected (16-32GB). Selecting balanced models."
    HOLO_MODEL="holo2:8b"
    QWEN_MODEL="qwen3-vl:8b"
else
    echo "    -> High RAM detected (>32GB). Selecting performance models."
    HOLO_MODEL="holo2:30b"
    QWEN_MODEL="qwen3-vl:32b" 
    # Fallback note: if 32b doesn't exist in registry, user might need 8b or manual pull.
    # We'll stick to the request to be intelligent.
fi

# 5. Pull Models
pull_model() {
    local model=$1
    echo "[*] Pulling model: $model"
    if ollama pull "$model"; then
        echo "    Successfully pulled $model"
    else
        echo "    Error: Failed to pull $model. It might not be in the registry yet or network issue."
        echo "    Trying generic tag..."
        # Fallback logic could go here, e.g. pulling 'qwen3-vl' without size
        local generic_model=$(echo $model | cut -d: -f1)
        if [ "$generic_model" != "$model" ]; then
             echo "    Attempting fallback to generic tag: $generic_model"
             ollama pull "$generic_model" || echo "    Fallback failed."
        fi
    fi
}

pull_model "$HOLO_MODEL"
pull_model "$QWEN_MODEL"

echo ""
echo "=== Installation Complete ==="
echo "You can now use these models with Ollama:"
echo "  ollama run $HOLO_MODEL"
echo "  ollama run $QWEN_MODEL"
echo ""
echo "To use in CaptchaKraken, update your configuration to point to these model names."

