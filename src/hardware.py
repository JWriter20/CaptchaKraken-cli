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
        if "bf16" in force_model.lower():
            return (
                "Jake-Writer-Jobharvest/qwen3-vl-8b-merged-bf16",
                "Full BF16 (Forced)",
                "Using forced BF16 model (vLLM recommended)."
            )
        if "BF16" in force_model.lower() or "full" in force_model.lower():
            return (
                "Jake-Writer-Jobharvest/qwen3-vl-8b-merged-BF16",
                "Full BF16 (Forced)",
                "Using forced BF16 model (Transformers recommended)."
            )

    vram = get_gpu_memory_gb()
    device = get_device()
    is_mac = (device == "mps")

    # Full version: SAM3 (3.5) + Qwen3-VL (17.5) = 21GB
    # Requirements: 22GB VRAM or 32GB Unified Memory
    if (not is_mac and vram >= 22.0) or (is_mac and vram >= 32.0):
        # Prefer BF16 for vLLM performance if on CUDA
        if device == "cuda":
            return (
                "Jake-Writer-Jobharvest/qwen3-vl-8b-merged-bf16",
                "Full (Merged BF16)",
                "Hardware supports high-performance BF16 models via vLLM."
            )
        return (
            "Jake-Writer-Jobharvest/qwen3-vl-8b-merged-BF16",
            "Full (Merged BF16)",
            "Hardware supports full precision models."
        )

    # Insufficient hardware for local execution of the full merged model
    return (
        "API",
        "None (Insufficient Hardware)",
        "Insufficient hardware detected for local execution (22GB+ VRAM required). "
        "To force local execution, set CAPTCHA_FORCE_MODEL=bf16 (for vLLM) or BF16."
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

        version_msg = f"PyTorch {torch.__version__}"
        if "rocm" in torch.__version__.lower():
            version_msg += " (ROCm build)"
        elif "cuda" in torch.__version__.lower():
            version_msg += " (CUDA build)"
        messages.append(version_msg)

        # Check GPU
        device = get_device()
        gpu_type = get_gpu_type()
        
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            messages.append(f"{gpu_type} available: {gpu_name} ({gpu_mem:.1f}GB)")
        elif device == "mps":
            messages.append(f"{gpu_type} available")
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

    # Add model recommendation
    _, model_desc, model_msg = get_recommended_model_config()
    messages.append(f"Recommendation: {model_desc} - {model_msg}")

    return success, " | ".join(messages)


def get_device() -> str:
    """
    Get the best available device for inference.
    Correctly detects CUDA (NVIDIA), MPS (Apple Silicon), or CPU.
    """
    try:
        import torch

        if torch.cuda.is_available():
            # Check if it's actually ROCm
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                return "cuda" # Still use 'cuda' as device string, but we know it's ROCm/HIP
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_gpu_type() -> str:
    """
    Returns a string describing the GPU type: "NVIDIA", "AMD (ROCm)", "Apple (MPS)", or "None".
    """
    try:
        import torch
        if torch.cuda.is_available():
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                return "AMD (ROCm)"
            return "NVIDIA (CUDA)"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple (MPS)"
    except ImportError:
        pass
    return "None"


if __name__ == "__main__":
    ok, msg = check_requirements()
    print(f"Requirements check: {'PASS' if ok else 'FAIL'}")
    print(msg)

