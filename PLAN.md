# CaptchaKraken Implementation Plan

## Overview
We are building a library to solve captchas by parsing images into interactive components and using an LLM to generate solution steps. We support both cloud-based LLMs (like OpenAI) and specialized local models.

**Core Workflow:**
1.  **Input:** Captcha Image (path or bytes).
2.  **Parsing:** Use `OmniParser` to detect clickable elements (bounding boxes, labels). *Note: OmniParser is optional if using Holo1.5.*
3.  **Reasoning:** Send image + parsed elements + user prompt (e.g., "Click traffic lights") to an LLM.
4.  **Output:** JSON sequence of actions (Click, Drag, Type).

## Detailed Roadmap

### Phase 1: Python Core & Environment Setup
- [ ] **Project Initialization**
    - Create `requirements.txt` / `pyproject.toml`.
    - Setup Virtual Environment (`venv`).
    - directory structure: `src/captchakraken`, `tests`, `examples`.
- [ ] **Dependency Installation**
    - Install `torch`, `torchvision` (for OmniParser/Local Models).
    - Install `openai` (or other LLM client).
    - Install `pillow` for image handling.
    - Clone/Setup `OmniParser` (Microsoft) if available as repo, or download weights.

### Phase 2: OmniParser Integration (The "Eyes")
- [ ] **Parser Module (`src/captchakraken/parser.py`)**
    - Create `CaptchaParser` class.
    - Implement `parse(image_path) -> List[Component]`.
    - `Component` dataclass: `{id, label, box: [x1, y1, x2, y2], confidence}`.
- [ ] **Validation Script**
    - Create `scripts/test_parser.py`.
    - Run against all images in `captchaimages/`.
    - Visualize results (draw boxes on images) to verify accuracy.

### Phase 3: LLM Integration (The "Brain")
- [ ] **Solver Module (`src/captchakraken/solver.py`)**
    - Create `CaptchaSolver` class.
    - Define Action Schema (JSON): `[{action: 'click', target_id: 1}, {action: 'drag', from: 1, to: 2}]`.
    - Implement `generate_solution(components, task_description)`.
    - Construct Prompt: "Given these UI elements with IDs, generate steps to [Task]...".
- [ ] **Strategy Selection & Local Model Support**
    - Implement a Strategy Pattern to switch between solvers:
        - **Strategy A: Holo1.5 (End-to-End)**
            - *Model:* Holo1.5 (7B) based on Qwen2.5-VL.
            - *Flow:* Image -> Model -> Action (Coords). No OmniParser needed.
            - *Pros:* Faster, specialized for UI.
        - **Strategy B: OmniParser + Qwen3 (Two-Step)**
            - *Model:* Qwen_Qwen3-8B or Qwen_Qwen3-14B (depending on VRAM).
            - *Flow:* Image -> OmniParser -> Parsed Elements -> Qwen3 -> Action (ID-based).
            - *Pros:* Potentially higher accuracy on complex reasoning; modular.
    - **Requirement:** Local installation required for both strategies.
- [ ] **LLM Client**
    - Support OpenAI API initially.
    - Environment variable `OPENAI_API_KEY`.

### Phase 4: Javascript/Typescript Wrapper
- [ ] **Node.js Package Structure**
    - `npm init`.
    - `index.ts` wrapper.
- [ ] **Python-JS Bridge**
    - Option A: Python runs as a CLI tool, Node spawns process. (Recommended for simplicity).
    - Option B: Python runs a local server.
    - Implement `execFile` / `spawn` in Node to call Python solver.
- [ ] **Typed Interfaces**
    - Define TS interfaces for `CaptchaAction`, `CaptchaComponent`.

### Phase 5: Testing & Polishing
- [ ] **End-to-End Tests**
    - Python tests: `tests/test_solver.py`.
    - JS tests: `test/integration.test.ts`.
- [ ] **Documentation**
    - Usage guide in README.
    - Example scripts.

## Test Data
Images in `captchaimages/`:
- Cloudflare Turnstile
- hCaptcha (Basic, Drag)
- reCAPTCHA (Basic, Grid)

## Action Schema

The solver must return a JSON array of action objects. Each object must strictly adhere to one of the following structures:

### 1. Click
Used for clicking a specific element or coordinate.
```json
{
  "action": "click",
  "target_id": number | null,       // ID of the component (if parsed)
  "coordinates": [x, y] | null      // [x, y] coordinates (if no ID or direct coord click)
}
```
*Constraint:* At least one of `target_id` or `coordinates` must be present.

### 2. Drag
Used for drag-and-drop captchas (e.g., puzzle pieces).
```json
{
  "action": "drag",
  "source_id": number | null,       // ID of the item to drag
  "source_coordinates": [x, y] | null,
  "target_id": number | null,       // ID of the drop zone
  "target_coordinates": [x, y] | null
}
```

### 3. Type
Used for text-based captchas or input fields.
```json
{
  "action": "type",
  "text": string,
  "target_id": number | null        // Optional: click this ID before typing
}
```

### 4. Wait
Used if a delay is needed between actions (e.g., waiting for an animation).
```json
{
  "action": "wait",
  "duration_ms": number
}
```

### 5. Request updated image
Used if the image needs to be updated before the next action.
```json
{
  "action": "request_updated_image" // For google images that slowly fade into the next image, after clicking the first round of images, we need to wait for the image to update before clicking the next round of images.
}
```