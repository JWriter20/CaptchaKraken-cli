# Model Decisions Analysis (vLLM Provider)

This document explains the calls and decisions made by the `Jake-Writer-Jobharvest/qwen3-vl-8b-merged-bf16` model running via `vLLM` during the solver tests.

## 1. Grid Selection (reCAPTCHA Style)

For reCAPTCHA-style grids, the solver uses `find_grid` to identify the cells and then overlays numbers (1-9 or 1-16). The planner then decides which numbers to click.

### test_3x3_recaptcha
- **Media**: `captchaimages/coreRecaptcha/recaptchaImages.png`
- **Detected**: 3x3 grid (9 cells).
- **Model Decision**: Final selection: `[2, 4, 6, 9]`.
- **Reasoning**: The model identifies the tiles matching the target description accurately.

### test_4x4_recaptcha
- **Media**: `captchaimages/coreRecaptcha/recaptchaImages2.png`
- **Detected**: 4x4 grid (16 cells).
- **Model Decision**: Final selection: `[2, 3, 4, 6, 7, 8, 10, 11, 12, 15, 16]`.
- **Reasoning**: The model correctly identifies multiple tiles that form a larger object, respecting the 4x4 grid structure.

### test_slanted_grid
- **Media**: `captchaimages/slantedGrid.png`
- **Detected**: 3x3 grid (9 cells).
- **Model Decision**: Final selection: `[4, 8, 9]`.
- **Reasoning**: The model handles slanted perspective distortion and identifies the correct cells.

## 2. General Solving (hCaptcha Style)

For hCaptcha and other non-grid puzzles, the planner uses tools like `detect` or `simulate_drag`.

### test_hcaptcha_puzzle_solve
- **Media**: `captchaimages/hcaptchaPuzzle2.png`
- **Goal identified**: "click on the two shapes that are most similar to each other"
- **Model Decision**: Action: `click` on "a green cube with a grid pattern", `max_items`: 2.
- **Reasoning**: The model correctly identifies that **two** items are needed and sets `max_items: 2`. This allows the `detect` tool to return both similar shapes.

### test_hcaptcha_choose_similar_shapes
- **Media**: `captchaimages/hcaptchaChooseSimilarShapes.png`
- **Goal identified**: "click on the 2D shape that looks like a cube"
- **Model Decision**: Action: `click` on "a 3d cube", `max_items`: 2.
- **Reasoning**: The model identifies the target shapes and uses `max_items: 2` for the similarity matching task.

## 3. Drag Puzzles (Iterative Refinement)

For drag puzzles, the model identifies the small draggable object and then iteratively refines its position within the main image area.

### Simplification: Removal of Target Hints
We have removed the `target_hint` parameter from `simulate_drag`. The model now only provides the `source` description. The destination is automatically inferred by the `simulate_drag` tool using the main `goal`/instruction context, with a **Position-based Role Correction** logic that ensures the "source" is the piece in the corner/extreme box and the "target" is in the central area.

### test_hcaptcha_drag_image_1
- **Media**: `captchaimages/hcaptchaDragImage1.png`
- **Source identified**: "pink segment"
- **Tool Call**: `simulate_drag(source='pink segment')`
- **Detection**: Source detected at `(0.899, 0.947)` (bottom-right). Target spot detected at `(0.520, 0.273)` (center).
- **Process**: Refinement iterations adjusted the position based on visual overlays.

### test_hcaptcha_drag_images_3
- **Media**: `captchaimages/hcaptchaDragImages3.png`
- **Source identified**: "the bee"
- **Tool Call**: `simulate_drag(source='the bee')`
- **Detection**: Source detected at `(0.135, 0.326)` (top-left). Target spot (strawberry) detected at `(0.897, 0.945)` (bottom-right destination area).
- **Process**: The model correctly moved the bee towards the strawberry.

## 4. Video Captchas

### test_hcaptcha_video_webm
- **Media**: `captchaimages/hcaptcha_1766539373078.webm`
- **Process**: First frame extracted.
- **Detected**: 3x3 grid.
- **Model Decision**: Final selection: `[5]`.
- **Reasoning**: The model identifies the central cell as the target for the video's instruction.

---
*Note: All tests were run using vLLM with `gpu_memory_utilization=0.65` and SAM 3 for high-accuracy grounding/segmentation.*
