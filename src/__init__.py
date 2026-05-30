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
from .planner import ActionPlanner
from .solver import CaptchaSolver, solve_captcha

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
