# Based on user-provided specification for RTX 5090
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 1. Install vLLM from source (User's request for 5090 support)
RUN git clone https://github.com/vllm-project/vllm.git --depth 1
WORKDIR /app/vllm
RUN pip install --upgrade pip
RUN pip install -r requirements/build.txt
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN pip install -e .

WORKDIR /app

# 2. Install CaptchaKraken-cli dependencies
COPY requirements.txt .
# Remove vllm from requirements.txt since we installed from source
RUN sed -i '/vllm/d' requirements.txt && pip install -r requirements.txt
# Add server dependencies
RUN pip install fastapi uvicorn huggingface_hub

# 3. Copy Code
COPY . /app/CaptchaKraken-cli
ENV PYTHONPATH="${PYTHONPATH}:/app/CaptchaKraken-cli"

# 4. Bake in Models
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV SAM3_MODEL_ID="Jake-Writer-Jobharvest/sam3"
ENV MODEL="Qwen/Qwen3-VL-8B-Instruct-FP8"

# Download Models
# Using standard HF cache ~/.cache/huggingface/hub
RUN python3 -c "import os; from huggingface_hub import snapshot_download; \
    token = os.getenv('HF_TOKEN'); \
    print('Downloading SAM3...'); snapshot_download(os.getenv('SAM3_MODEL_ID'), token=token); \
    print('Downloading Base Model...'); snapshot_download(os.getenv('MODEL'), token=token); \
    print('Downloading LoRAs...'); \
    snapshot_download('Jake-Writer-Jobharvest/qwen3-vl-8b-general-lora', token=token); \
    snapshot_download('Jake-Writer-Jobharvest/qwen3-vl-8b-grid-lora', token=token);" || echo "Warning: Model download failed. Models will be downloaded at runtime."

# 5. Environment Variables
ENV CAPTCHA_PLANNER_BACKEND="vllm"
ENV VLLM_GPU_MEMORY_UTILIZATION="0.75"
ENV PORT=8000

# 6. Entrypoint
# We use the docker_server.py as the entrypoint
WORKDIR /app/CaptchaKraken-cli
CMD ["python3", "src/docker_server.py"]
