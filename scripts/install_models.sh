#!/bin/bash

# scripts/install_models.sh
# Installs dependencies for CaptchaKraken's two-stage architecture:
# 1. Action Planner: Uses Ollama or Gemini for action planning
# 2. Attention Extractor: Uses small VLMs via transformers for coordinate extraction

set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "=== CaptchaKraken Setup ==="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 1. Detect OS
OS="$(uname -s)"

# 2. Setup Virtual Environment
echo "[1/4] Setting up Python environment..."

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "    No active virtual environment detected."
    
    VENV_PATH="$PROJECT_ROOT/venv"
    
    if [ ! -d "$VENV_PATH" ]; then
        echo "    Creating virtual environment at $VENV_PATH..."
        python3 -m venv "$VENV_PATH"
    fi
    
    echo "    Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "    Using active virtual environment: $VIRTUAL_ENV"
fi

# 3. Install Python dependencies
echo ""
echo "[2/4] Installing Python packages..."

if command_exists pip; then
    pip install --upgrade pip
    
    # Detect GPU type and install appropriate PyTorch
    if command_exists nvidia-smi; then
        echo "    NVIDIA GPU detected, installing CUDA PyTorch..."
        pip install torch torchvision
    elif command_exists rocm-smi || [ -d "/opt/rocm" ]; then
        echo "    AMD GPU detected, installing ROCm PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
    elif [ "$OS" = "Darwin" ]; then
        echo "    macOS detected, installing MPS-enabled PyTorch..."
        pip install torch torchvision
    else
        echo "    No GPU detected, installing CPU PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install core dependencies (Ollama + Gemini backends)
    pip install transformers accelerate pillow pydantic numpy scipy ollama google-genai
    echo "    ✓ Python packages installed"
else
    echo "    Error: 'pip' not found."
    exit 1
fi

# 4. Setup Ollama (optional but recommended)
echo ""
echo "[3/4] Checking Ollama setup..."

if command_exists ollama; then
    echo "    ✓ Ollama is installed"
    
    # Check if ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "    ✓ Ollama server is running"
        
        # Pull recommended models
        echo "    Pulling recommended models (this may take a while)..."
        
        echo "    → Pulling gemma-3 (action planning)..."
        ollama pull hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M 2>/dev/null || echo "      ⚠ Could not pull gemma-3, you can do this manually"
    else
        echo "    ⚠ Ollama is installed but not running."
        echo "      Start it with: ollama serve"
    fi
else
    echo "    ⚠ Ollama is not installed."
    echo "      For local inference, install from: https://ollama.ai"
    echo "      Alternatively, use OpenAI API by setting OPENAI_API_KEY"
fi

# 5. Verify installation
echo ""
echo "[4/4] Verifying installation..."

cd "$PROJECT_ROOT"
python3 -c "
import sys
print(f'Python {sys.version}')

try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
    print(f'PyTorch {torch.__version__} ({device})')
except ImportError:
    print('PyTorch: NOT INSTALLED')
    
try:
    import transformers
    print(f'Transformers {transformers.__version__}')
except ImportError:
    print('Transformers: NOT INSTALLED')

try:
    import ollama
    models = ollama.list()
    count = len(models.get('models', []))
    print(f'Ollama: {count} models available')
except:
    print('Ollama: NOT AVAILABLE')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  # Single step"
echo "  python -m src.captchakraken.cli captcha.png 'Solve this captcha'"
echo ""
echo "  # Python API"
echo "  from src import CaptchaSolver"
echo "  solver = CaptchaSolver()"
echo "  action = solver.solve_step('captcha.png', 'Solve this captcha')"
echo ""
echo "Configuration:"
echo "  --planner ollama|gemini      Backend for action planning"
echo "  --attention-model MODEL      HuggingFace model for coordinate extraction"
echo ""
