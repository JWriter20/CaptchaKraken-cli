"""
Hardware detection utilities for CaptchaKraken.

Checks system capabilities for running local models.
"""

import sys
from typing import Tuple


def check_requirements() -> Tuple[bool, str]:
    """
    Check if system meets requirements for local model inference.

    Returns:
        Tuple of (success: bool, message: str)
    """
    messages = []
    success = True

    # Check Python version
    if sys.version_info < (3, 8):
        success = False
        messages.append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")

    # Check PyTorch
    try:
        import torch

        messages.append(f"PyTorch {torch.__version__}")

        # Check CUDA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            messages.append(f"CUDA available: {gpu_name} ({gpu_mem:.1f}GB)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            messages.append("MPS (Apple Silicon) available")
        else:
            messages.append("No GPU detected - CPU inference will be slower")

    except ImportError:
        success = False
        messages.append("PyTorch not installed")

    # Check transformers
    try:
        import transformers

        messages.append(f"Transformers {transformers.__version__}")
    except ImportError:
        success = False
        messages.append("Transformers not installed")

    # Check ollama connectivity
    try:
        import ollama

        models = ollama.list()
        model_count = len(models.get("models", []))
        messages.append(f"Ollama connected ({model_count} models)")
    except Exception as e:
        messages.append(f"Ollama not available: {e}")

    return success, " | ".join(messages)


def get_device() -> str:
    """
    Get the best available device for inference.

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_gpu_memory_gb() -> float:
    """
    Get available GPU memory in GB.

    Returns:
        GPU memory in GB, or 0 if no GPU
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        pass
    return 0.0


if __name__ == "__main__":
    ok, msg = check_requirements()
    print(f"Requirements check: {'PASS' if ok else 'FAIL'}")
    print(msg)
