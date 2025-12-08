"""
CaptchaKraken - AI-powered captcha solver using LLM planning + vision tools.

Usage:
    from src import CaptchaSolver

    # Gemini (default cloud backend)
    solver = CaptchaSolver(
        provider="gemini",
        api_key="your-gemini-api-key"
    )

    # Ollama (local backend)
    solver = CaptchaSolver(provider="ollama")

    # Solve a captcha
    actions = solver.solve("captcha.png", "Select all traffic lights")
"""

from pathlib import Path

# Best-effort load of .env from project root so GEMINI_API_KEY and others are
# available even when running scripts directly (e.g. scripts/test_detection.py).
try:  # pragma: no cover - environment setup
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
except Exception:
    # If python-dotenv is not installed or .env is missing, just continue.
    pass

from .action_types import (
    CaptchaAction,
    ClickAction,
    DragAction,
    TypeAction,
    WaitAction,
)
from .attention import AttentionExtractor
from .imagePreprocessing import get_grid_bounding_boxes
from .overlay import add_drag_overlay, add_overlays_to_image
from .planner import ActionPlanner
from .solver import CaptchaSolver, solve_captcha

__all__ = [
    # Main solver
    "CaptchaSolver",
    "solve_captcha",
    # Components
    "ActionPlanner",
    "AttentionExtractor",
    # Action Types
    "CaptchaAction",
    "ClickAction",
    "DragAction",
    "TypeAction",
    "WaitAction",
    # Utilities
    "add_overlays_to_image",
    "add_drag_overlay",
    "get_grid_bounding_boxes",
]

__version__ = "0.4.0"
