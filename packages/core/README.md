# CaptchaKraken
AI-powered captcha solver using attention-based coordinate extraction.

## Overview
CaptchaKraken solves captchas using a two-stage approach:

1. **Action Planning** (Ollama/OpenAI API) - An LLM analyzes the captcha to determine what action is needed
2. **Attention-Based Targeting** (Transformers) - A local VLM processes the target, and we extract **actual attention weights** from the model's internal layers to find coordinates

This separates "what to do" from "where to do it", using real attention values (not prompting for coordinates).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1: ActionPlanner (Ollama/OpenAI API)                             │
│  ─────────────────────────────────────────                              │
│  Input:  Image + "Solve this captcha"                                   │
│  Output: action_type="click", target="the checkbox"                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 2: AttentionExtractor (Transformers - Local Model)               │
│  ────────────────────────────────────────────────────────               │
│  1. Load moondream2 (or other VLM) via transformers                     │
│  2. Pass image + "the checkbox" through model                           │
│  3. Hook into attention layers, capture weight tensors                  │
│  4. Analyze attention matrix: which image patches get highest weights?  │
│  5. Map patch grid position → pixel coordinates                         │
│  Output: coordinates=[342, 156]                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## How Attention Extraction Works

```
Image (400x300) → Vision Encoder → 14×14 grid = 196 image patch tokens

Text: "the checkbox" → Text tokens

                    ATTENTION MATRIX
                    ────────────────
                    Patch_0  Patch_1  Patch_2  ... Patch_195
    "the"           [0.01,   0.02,    0.01,   ...  0.01]
    "checkbox"      [0.01,   0.85,    0.02,   ...  0.01]
                             ↑
                    HIGH ATTENTION = this patch contains the checkbox
                    Grid position (0, 1) → Pixel coordinates (21, 10)
```

We extract the **raw attention weight tensors** from transformer layers, not the model's text output.

## Installation

### Prerequisites
- Python 3.8+
- CUDA GPU recommended (for attention model)
- Ollama installed (for action planning)

### Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Ollama and pull the planning model
# https://ollama.ai
ollama pull hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M

# The attention model (moondream2) downloads automatically on first run
```

## Usage

### Python - Single Step
```python
from src.captchakraken import CaptchaSolver

solver = CaptchaSolver()

# Get the next action to solve the captcha
action = solver.solve_step("captcha.png", "Solve this captcha")
print(action)
# ClickAction(action='click', coordinates=[342, 156])
```

### Python - Full Loop
```python
from src.captchakraken import CaptchaSolver

solver = CaptchaSolver()

def get_current_image():
    return screenshot_captcha()

def is_captcha_gone():
    return not captcha_element.is_visible()

for action in solver.solve_loop(
    get_image=get_current_image,
    context="Solve this captcha",
    max_steps=10,
    end_condition=is_captcha_gone
):
    execute_action(action)
```

### CLI
```bash
# Single step
python -m src.captchakraken.cli captcha.png "Solve this captcha"

# Use OpenAI for planning instead of Ollama
python -m src.captchakraken.cli captcha.png "Click the checkbox" --planner openai
```

## Configuration

### Environment Variables
```bash
# For OpenAI planning (optional, default is Ollama)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Solver Options
```python
solver = CaptchaSolver(
    planner_backend="ollama",                                # "ollama" or "openai"
    planner_model="hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M", # Model for action planning
    attention_model="vikhyatk/moondream2",                   # HuggingFace model for attention
    device="cuda",                                           # "cuda", "cpu", or "mps"
)
```

## Action Types

| Action | Description | Fields |
|--------|-------------|--------|
| `click` | Click at coordinates | `coordinates: [x, y]` |
| `drag` | Drag from source to target | `source_coordinates`, `target_coordinates` |
| `type` | Type text | `text: str` |
| `wait` | Wait before next action | `duration_ms: int` |
| `request_updated_image` | Need fresh screenshot | - |

Coordinates are returned as percentages in the range `[0.0, 1.0]` where `(0.0, 0.0)` is the top-left corner and `(1.0, 1.0)` is the bottom-right corner.

## Project Structure

```
src/captchakraken/
├── solver.py      # Main CaptchaSolver - orchestrates both stages
├── planner.py     # Stage 1: ActionPlanner (Ollama/OpenAI)
├── attention.py   # Stage 2: AttentionExtractor (transformers)
├── types.py       # Pydantic action models
└── cli.py         # Command-line interface
```

## License
MIT
