"""
CaptchaKraken - AI-powered captcha solver using attention-based coordinate extraction.

Usage:
    from src.captchakraken import CaptchaSolver
    
    solver = CaptchaSolver()
    action = solver.solve_step("captcha.png", "Solve this captcha")
"""

from .solver import CaptchaSolver, solve_captcha
from .planner import ActionPlanner, PlannedAction
from .attention import AttentionExtractor
from .types import (
    CaptchaAction,
    ClickAction,
    DragAction,
    TypeAction,
    WaitAction,
    RequestUpdatedImageAction,
    Solution,
)

__all__ = [
    # Main solver
    "CaptchaSolver",
    "solve_captcha",
    
    # Components
    "ActionPlanner",
    "PlannedAction",
    "AttentionExtractor",
    
    # Types
    "CaptchaAction",
    "ClickAction",
    "DragAction",
    "TypeAction",
    "WaitAction",
    "RequestUpdatedImageAction",
    "Solution",
]

__version__ = "0.2.0"

