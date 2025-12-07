# 🐙 CaptchaKraken

**Next-Gen AI Captcha Solver**

CaptchaKraken is a modular, AI-powered captcha solving ecosystem that uses advanced Vision Language Models (VLMs) and LLMs to solve complex captchas like hCaptcha, reCAPTCHA, and Cloudflare Turnstile.

## 🚀 Features

- **Multi-Modal AI**: Uses local VLMs (Moondream2, etc.) for precise coordinate extraction and LLMs (Ollama/OpenAI) for high-level planning.
- **Two-Stage Solving**: Separates "what to do" (Action Planning) from "where to do it" (Attention Extraction).
- **Iterative Refinement**: Self-correcting visual feedback loop for drag-and-drop puzzles.
- **Monorepo Structure**: Isolated packages for Core logic, Python Playwright integration, and TypeScript Playwright integration.

## 📦 Packages

### 1. [Core](packages/core) (`packages/core`)
The brain of the operation. Contains the `CaptchaSolver`, `ActionPlanner`, and `AttentionExtractor`.
- **Python**: `captchakraken`

### 2. [Playwright Python](packages/playwright-py) (`packages/playwright-py`)
Seamless integration for Python Playwright automation.
- **Install**: `pip install captchakraken-playwright`

### 3. [Playwright TypeScript](packages/playwright-js) (`packages/playwright-js`)
Seamless integration for TypeScript/JavaScript Playwright automation.
- **Install**: `npm install captchakraken-playwright`

## 🛠️ Development

### Prerequisites
- Python 3.8+
- Node.js 18+
- [Ollama](https://ollama.ai) (for local LLM inference)

### Setup
1. Clone the repository
2. Install dependencies for core:
   ```bash
   cd packages/core
   pip install -r requirements.txt
   ```
3. Pull the required models:
   ```bash
   ollama pull hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M
   ```

### Running Tests
We provide scripts to test the detection and pointing capabilities of the core model.

```bash
# Test object detection
python packages/core/scripts/test_detection.py path/to/image.png "car"

# Test precise pointing
python packages/core/scripts/test_pointing.py path/to/image.png "the checkbox"
```

Results are saved to the `test-results/` directory.

## 📄 License
MIT
