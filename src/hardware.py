"""
Hardware detection utilities for CaptchaKraken.

Checks system capabilities for running local models.
"""

import sys
import os
import subprocess
from typing import Tuple, Optional, Literal


def get_total_system_memory_gb() -> float:
    """
    Get total system RAM in GB. Useful for Mac Unified Memory detection.
    """
    try:
        if sys.platform == "darwin":
            # macOS
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
            return mem_bytes / (1024**3)
        elif sys.platform == "linux":
            # Linux
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_kb = int(line.split()[1])
                        return mem_kb / (1024**2)
    except Exception:
        pass
    return 0.0


def get_gpu_memory_gb() -> float:
    """
    Get available GPU memory in GB.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # For Apple Silicon, we use total system memory as "unified memory"
            return get_total_system_memory_gb()
    except ImportError:
        pass
    return 0.0


def get_recommended_model_config() -> Tuple[str, str, str]:
    """
    Determine which model configuration to use based on hardware.

    Returns:
        Tuple of (planner_model, description, recommendation_msg)
    """
    # Allow overrides via environment variables
    force_model = os.getenv("CAPTCHA_FORCE_MODEL")
    if force_model:
        if "q8" in force_model.lower():
            return (
                "Jake-Writer-Jobharvest/qwen3-vl-8b-merged-q8",
                "Quantized (Forced)",
                "Using forced quantized model."
            )
        elif "fp16" in force_model.lower() or "full" in force_model.lower():
            return (
                "Jake-Writer-Jobharvest/qwen3-vl-8b-merged-fp16",
                "Full (Forced)",
                "Using forced full model."
            )

    vram = get_gpu_memory_gb()
    device = get_device()
    is_mac = (device == "mps")

    # Full version: SAM3 (3.5) + FP16 (17.5) = 21GB
    # Requirements: 22GB VRAM or 32GB Unified Memory
    if (not is_mac and vram >= 22.0) or (is_mac and vram >= 32.0):
        return (
            "Jake-Writer-Jobharvest/qwen3-vl-8b-merged-fp16",
            "Full",
            "Hardware supports full precision models."
        )

    # Quantized version: SAM3 (3.5) + Q8 (10) = 13.5GB
    # Requirements: 15GB VRAM or 30GB Unified Memory
    if (not is_mac and vram >= 15.0) or (is_mac and vram >= 30.0):
        return (
            "Jake-Writer-Jobharvest/qwen3-vl-8b-merged-q8",
            "Quantized (Q8)",
            "Hardware supports quantized models."
        )

    # Insufficient hardware
    return (
        "API",
        "None (Insufficient Hardware)",
        "Insufficient hardware detected. Recommended: Get an API token at captchaKraken.com or use a cloud provider. "
        "To force local execution anyway, set CAPTCHA_FORCE_MODEL=q8 or full."
    )


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

    # Add model recommendation
    _, model_desc, model_msg = get_recommended_model_config()
    messages.append(f"Recommendation: {model_desc} - {model_msg}")

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


if __name__ == "__main__":
    ok, msg = check_requirements()
    print(f"Requirements check: {'PASS' if ok else 'FAIL'}")
    print(msg)

