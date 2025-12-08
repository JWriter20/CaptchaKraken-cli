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
