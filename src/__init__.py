"""
CaptchaKraken - AI-powered captcha solver using attention-based coordinate extraction.

Usage:
    from src import CaptchaSolver
    
    # Ollama (default)
    solver = CaptchaSolver(
        model="llama3.2:3b",
        provider="ollama"
    )
    
    # OpenAI
    solver = CaptchaSolver(
        model="gpt-4o",
        provider="openai",
        api_key="your-api-key"
    )
    
    action = solver.solve_step("captcha.png")
"""

from .solver import CaptchaSolver, solve_captcha
from .planner import ActionPlanner, PlannedAction
from .attention import AttentionExtractor
from .action_types import (
    CaptchaAction,
    ClickAction,
    DragAction,
    TypeAction,
    WaitAction,
    RequestUpdatedImageAction,
    VerifyAction,
    Solution,
)
from .overlay import add_overlays_to_image

__all__ = [
    # Main solver
    "CaptchaSolver",
    "solve_captcha",
    
    # Components
    "ActionPlanner",
    "PlannedAction",
    "AttentionExtractor",
    
    # Action Types
    "CaptchaAction",
    "ClickAction",
    "DragAction",
    "TypeAction",
    "WaitAction",
    "RequestUpdatedImageAction",
    "VerifyAction",
    "Solution",
    
    # Utilities
    "add_overlays_to_image",
]

__version__ = "0.3.0"

