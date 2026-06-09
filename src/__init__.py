"""
CaptchaKraken v2 — vLLM-backed captcha solver.

The v1 implementation (SAM3 grounding, multiple LLM providers, detect/segment
tool calls, drag refinement) lives on the `v1-old-architecture` branch in both
this repo and the parent PlaywrightCaptchaKrakenJS repo. Grab files from there
if you need to bring back simulate_drag / segment / detect for action
verification work later.

Usage:
    from src import CaptchaSolver
    solver = CaptchaSolver()  # talks to local vLLM, model='captcha' LoRA
    actions = solver.solve("captcha.png")
"""

from pathlib import Path

try:  # pragma: no cover
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
except Exception:
    pass

from .action_types import (
    CaptchaAction,
    ClickAction,
    DragAction,
    TypeAction,
    WaitAction,
)
from .image_processor import ImageProcessor
from .overlay import add_overlays_to_image

# The planner (requests) and solver (torch/vllm/transformers) pull in the heavy
# serving stack. Keep them optional so leaf modules — e.g. tool_calls.find_grid,
# which needs only cv2 + numpy + pillow — can be imported in a minimal env (CI's
# hermetic grid-detection test) without the full GPU dependency set installed.
try:  # pragma: no cover - exercised only when the serving stack is installed
    from .planner import ActionPlanner
    from .solver import CaptchaSolver, solve_captcha
except ModuleNotFoundError:
    ActionPlanner = None  # type: ignore[assignment,misc]
    CaptchaSolver = None  # type: ignore[assignment,misc]
    solve_captcha = None  # type: ignore[assignment]

__all__ = [
    "CaptchaSolver",
    "solve_captcha",
    "ActionPlanner",
    "ImageProcessor",
    "CaptchaAction",
    "ClickAction",
    "DragAction",
    "TypeAction",
    "WaitAction",
    "add_overlays_to_image",
]

__version__ = "2.0.0"
