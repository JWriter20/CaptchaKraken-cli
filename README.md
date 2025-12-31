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

## Captcha support status

- [x] **Checkbox captchas** – end‑to‑end solving working.
- [x] **Image selection / image grid captchas** – end‑to‑end solving working.
- [ ] **Text captchas** – basic plumbing present, solving still in progress.

Additional captcha types and more robust classification/solving strategies are under active development.
