## CaptchaKraken CLI

AI-powered, fully local captcha-solving CLI that uses attention-based vision models to extract precise bounding boxes for common web captchas.

## Description

`CaptchaKraken` takes a screenshot of a captcha challenge, classifies the captcha type, highlights and numbers all interactable regions, and then plans the sequence of clicks needed to solve it.  
It is designed to be:

- **CLI-first**: run end‑to‑end solves from the command line.
- **Model-agnostic**: pluggable attention models for coordinate extraction.
- **Debuggable**: optional overlays and debug images to inspect detection and planning.

High-level flow:
1. **Classify** the captcha (checkbox vs image grid vs text prompt, etc.).
2. **Detect and number** all interactable elements in the captcha (checkboxes, tiles, buttons).
3. **Plan actions** using the detect and segmentation tools to generate action bounding boxes.
4. **Output** the sequence of actions (clicks) that can be replayed in a browser automation stack.

## Getting Started

### Prerequisites

- **Docker**: Required to run the local inference server (SAM 3 + vLLM).
- **GPU**: NVIDIA GPU with 22GB+ VRAM (e.g., RTX 3090/4090/5090) is highly recommended for local execution.

### Docker Setup

The `CaptchaKraken-cli` requires a running inference container to handle vision tasks (detection) and planning tasks (LLM).

1. **Build the container**:
   ```bash
   cd PlaywrightCaptchaKrakenJS/CaptchaKraken-cli
   bash build_container.sh
   ```

2. **Run the container**:
   Start the server with an API key for security.
   ```bash
   docker run -d \
     --name captchakraken-vllm \
     --gpus all \
     --ipc=host \
     -p 8000:8000 \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     -e VLLM_API_KEY="your_secret_api_key" \
     captchakraken-vllm-5090
   ```

### Using the CLI

Once the container is running, configure the CLI to point to it:

```bash
export CAPTCHA_TOOL_SERVER="http://localhost:8000"
export VLLM_API_KEY="your_secret_api_key"

# Run a solve on a local image
python cli.py --image captchaimages/hcaptchaPuzzle.png --solve
```

## Examples

### 1. Simple Detection
Find specific objects in an image using SAM 3 through the server:
```bash
python cli.py --image captchaimages/hcaptchaPuzzle.png --detect "colored segment"
```

### 2. End-to-End Solve
The CLI will automatically classify the captcha, call the planner, and output the necessary click coordinates:
```bash
python cli.py --image captchaimages/coreRecaptcha/recaptchaImages.png --solve
```

### 3. Debug Mode
Visualize what the model sees:
```bash
python cli.py --image captchaimages/slantedGrid.png --solve --debug
```

## Captcha support status

- [x] **Checkbox captchas** – end‑to‑end solving working.
- [x] **Image selection / image grid captchas** – end‑to‑end solving working.
- [ ] **Text captchas** – basic plumbing present, solving still in progress.

Additional captcha types and more robust classification/solving strategies are under active development.
