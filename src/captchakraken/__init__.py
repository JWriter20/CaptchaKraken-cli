"""
CaptchaKraken - AI-powered captcha solver using attention-based coordinate extraction.

Usage:
    from captchakraken import CaptchaSolver
    
    solver = CaptchaSolver()
    action = solver.solve_step("captcha.png", "Solve this captcha")
    
    # Intelligent solving with element info
    action = solver.solve_step_intelligent(
        "captcha.png",
        context="Select all traffic lights",
        elements=[{"element_id": 1, "bbox": [0, 0, 100, 100]}, ...],
        prompt_text="Select all images with traffic lights"
    )
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
    CaptchaContext,
    InteractableElement,
    DetectedObject,
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
    
    # Context Types
    "CaptchaContext",
    "InteractableElement",
    "DetectedObject",
    
    # Utilities
    "add_overlays_to_image",
]

__version__ = "0.3.0"
