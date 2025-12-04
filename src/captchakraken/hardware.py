import sys
import platform
import subprocess
import shutil

def get_ram_size_gb():
    # Returns RAM in GB
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
            return int(out) / (1024**3)
        elif platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        # MemTotal:        16306536 kB
                        parts = line.split()
                        return int(parts[1]) / (1024**2)
        return 0
    except Exception:
        return 0

def get_vram_size_gb():
    # Returns VRAM in GB for Nvidia. 
    # Returns 0 if no GPU found or not Nvidia.
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]).strip()
            # If multiple GPUs, takes the first one?
            lines = out.decode('utf-8').split('\n')
            return int(lines[0]) / 1024
        except Exception:
            return 0
    return 0

def is_apple_silicon():
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["uname", "-m"]).strip().decode('utf-8')
            return out == "arm64"
        except:
            pass
    return False

def check_requirements():
    """
    Checks if the system meets the requirements for local LLM inference.
    Returns (bool, message).
    """
    ram_gb = get_ram_size_gb()
    vram_gb = get_vram_size_gb()
    is_unified = is_apple_silicon()
    
    # Logic for model selection assumption:
    # We base requirements on the largest potential model for the tier.
    if ram_gb < 16:
        model_size_gb = 3.5 
    elif ram_gb < 32:
        model_size_gb = 6.0
    else:
        model_size_gb = 22.0

    # Check
    if is_unified:
        # Unified memory (Apple Silicon)
        # Requirement: At least half memory remains after loading
        required_ram = model_size_gb * 2
        if ram_gb >= required_ram:
            return True, f"Unified Memory: {ram_gb:.1f}GB >= 2*{model_size_gb}GB"
        else:
            return False, f"Unified Memory: {ram_gb:.1f}GB < 2*{model_size_gb}GB (Need at least {required_ram}GB RAM)"
            
    else:
        # Discrete GPU
        # Requirement: GPU can hold full model (VRAM >= Model)
        if vram_gb > 0:
            if vram_gb >= model_size_gb:
                 return True, f"VRAM: {vram_gb:.1f}GB >= {model_size_gb}GB"
            else:
                 return False, f"VRAM: {vram_gb:.1f}GB < {model_size_gb}GB"
        else:
            return False, "No capable Nvidia GPU detected (and not Unified Memory)."

    return False, "Unknown state"

