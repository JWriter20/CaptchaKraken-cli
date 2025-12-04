# CaptchaKraken
Locally runnable AI-based captcha solver.

## Overview
CaptchaKraken is a library designed to solve captchas by leveraging advanced Vision-Language Models (VLMs) and UI parsers. It supports a modular approach, allowing for different "eyes" (parsing) and "brains" (reasoning).

## Key Features
-   **Local & Cloud Support**: Use OpenAI for cloud-based reasoning or run specialized local models for complete privacy and control.
-   **OmniParser Integration**: Accurately detects interactive elements on captcha images.
-   **Structured Output**: Returns clean, executable JSON actions (e.g., `click`, `drag`).

## Supported Models

### ⋆ Holo2 - End-to-End
Holo2 is the most specialized open-weight model for UI interaction, fine-tuned specifically for "Computer Use" tasks.
-   **Base**: Qwen2.5-VL (fine-tuned)
-   **Flow**: Image -> Model -> Coords
-   **Specialty**: Trained on "Screenspot" and "WebClick".
-   **Advantage**: Natively outputs `[x, y]` coordinates. **No OmniParser needed.**

### ⋆ OmniParser + Qwen3 (8B/14B) - Two-Step
A robust two-step pipeline that separates detection from reasoning, often yielding better results on complex tasks depending on resources.
-   **Components**: OmniParser (for detection) + Qwen3-8B or Qwen3-14B (for reasoning).
-   **Flow**: Image -> OmniParser -> Bounding Boxes -> Qwen3 -> Action (ID selection).
-   **Advantage**: Leverages the strong reasoning capabilities of Qwen3 combined with the specialized detection of OmniParser. This modular approach allows for finer control.

**⚠️ Important Requirement**:
Both local strategies (Holo2 and OmniParser+Qwen3) require **local installation** and appropriate hardware (GPU). The Qwen3-14B model will require more VRAM than the 8B version or Holo2.

## Installation

### Prerequisites
-   Python 3.8+
-   Node.js (for the JS wrapper)
-   CUDA-capable GPU (highly recommended for local models)

### Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install Models**:
    Run the setup script to install Ollama and automatically download the appropriate Holo2 and Qwen3 models for your hardware:
    ```bash
    ./scripts/install_models.sh
    ```

3.  **Verify Installation**:
    Check that the models are available:
    ```bash
    ollama list
    ```

## Usage
(Coming soon)
