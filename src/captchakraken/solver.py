"""
CaptchaSolver - Main entry point for solving captchas.

This module orchestrates the two-stage solving process:
1. ActionPlanner - Determines what action to take (click, drag, wait, etc.)
2. AttentionExtractor - Finds where to perform the action using attention analysis

Usage:
    solver = CaptchaSolver()
    
    # Single step
    action = solver.solve_step("captcha.png", "Solve this captcha")
    
    # Full loop
    for action in solver.solve_loop(get_image, context, max_steps=10, end_condition=is_done):
        execute_action(action)
"""

import os
from typing import List, Optional, Callable, Iterator, Union
from pathlib import Path

from .types import (
    CaptchaAction,
    ClickAction,
    DragAction,
    TypeAction,
    WaitAction,
    RequestUpdatedImageAction,
)
from .planner import ActionPlanner, PlannedAction
from .attention import AttentionExtractor


class CaptchaSolver:
    """
    Two-stage captcha solver using LLM planning + attention-based coordinate extraction.
    
    Stage 1 (ActionPlanner): Ask an LLM (via Ollama or OpenAI API) what action is needed
    Stage 2 (AttentionExtractor): Extract attention weights from a local VLM to find coordinates
    
    Args:
        planner_backend: Backend for action planning ("ollama" or "openai")
        planner_model: Model for planning (default: "hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M" for ollama, "gpt-4o" for openai)
        attention_model: HuggingFace model for attention extraction (default: "vikhyatk/moondream2")
        openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        openai_base_url: Custom OpenAI base URL (or set OPENAI_BASE_URL env var)
        device: Device for attention model ("cuda", "cpu", "mps"). Auto-detected if None.
    """
    
    def __init__(
        self,
        planner_backend: str = "ollama",
        planner_model: Optional[str] = None,
        attention_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # Stage 1: Planner uses Ollama or OpenAI API
        self.planner = ActionPlanner(
            backend=planner_backend,
            model=planner_model,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
        )
        
        # Stage 2: Attention extractor uses transformers (local model)
        self.attention_extractor = AttentionExtractor(
            model=attention_model or "vikhyatk/moondream2",
            device=device,
        )
        
        # Track action history for context
        self._action_history: List[CaptchaAction] = []
    
    def solve_step(
        self,
        image_path: str,
        context: str = "",
        include_history: bool = True
    ) -> CaptchaAction:
        """
        Solve a single step of the captcha.
        
        This is the core method that:
        1. Asks the planner what action to take
        2. For spatial actions (click/drag), extracts coordinates via attention
        3. Returns a typed CaptchaAction object
        
        Args:
            image_path: Path to the captcha image
            context: Additional context (e.g., "This is an hCaptcha asking about traffic lights")
            include_history: Whether to include previous actions in the context
        
        Returns:
            A CaptchaAction object (ClickAction, DragAction, WaitAction, etc.)
        """
        # Resolve path
        image_path = str(Path(image_path).resolve())
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Build context with history
        full_context = context
        if include_history and self._action_history:
            history_str = self._format_history()
            full_context = f"{context}\n\nPrevious actions taken:\n{history_str}"
        
        # Stage 1: Plan the action
        print(f"[Stage 1] Planning next action...")
        planned = self.planner.plan(image_path, full_context)
        print(f"[Stage 1] Planned: {planned.action_type} - {planned.target_description or planned.reasoning}")
        
        # Convert to CaptchaAction based on type
        action = self._create_action(image_path, planned)
        
        # Track history
        self._action_history.append(action)
        
        return action
    
    def _create_action(self, image_path: str, planned: PlannedAction) -> CaptchaAction:
        """
        Convert a PlannedAction to a CaptchaAction, extracting coordinates if needed.
        """
        if planned.action_type == "click":
            # Stage 2: Extract coordinates via attention
            print(f"[Stage 2] Extracting click coordinates for: {planned.target_description}")
            x, y = self.attention_extractor.extract_coordinates(
                image_path,
                planned.target_description or "the target element"
            )
            print(f"[Stage 2] Coordinates: ({x}, {y})")
            return ClickAction(action="click", coordinates=[x, y])
        
        elif planned.action_type == "drag":
            # Extract both source and target coordinates
            print(f"[Stage 2] Extracting drag coordinates...")
            print(f"  Source: {planned.target_description}")
            print(f"  Target: {planned.drag_target_description}")
            
            source, target = self.attention_extractor.extract_drag_coordinates(
                image_path,
                planned.target_description or "the draggable element",
                planned.drag_target_description or "the drop target"
            )
            print(f"[Stage 2] Source: {source}, Target: {target}")
            
            return DragAction(
                action="drag",
                source_coordinates=list(source),
                target_coordinates=list(target)
            )
        
        elif planned.action_type == "type":
            return TypeAction(
                action="type",
                text=planned.text_to_type or ""
            )
        
        elif planned.action_type == "wait":
            return WaitAction(
                action="wait",
                duration_ms=planned.wait_duration_ms or 500
            )
        
        elif planned.action_type == "request_updated_image":
            return RequestUpdatedImageAction(action="request_updated_image")
        
        elif planned.action_type == "done":
            # Return a wait with 0 duration to signal completion
            # The caller should check for this or use the end_condition
            return WaitAction(action="wait", duration_ms=0)
        
        else:
            raise ValueError(f"Unknown action type: {planned.action_type}")
    
    def _format_history(self) -> str:
        """Format action history for context."""
        lines = []
        for i, action in enumerate(self._action_history[-5:], 1):  # Last 5 actions
            if isinstance(action, ClickAction):
                lines.append(f"{i}. Clicked at {action.coordinates}")
            elif isinstance(action, DragAction):
                lines.append(f"{i}. Dragged from {action.source_coordinates} to {action.target_coordinates}")
            elif isinstance(action, TypeAction):
                lines.append(f"{i}. Typed: '{action.text}'")
            elif isinstance(action, WaitAction):
                lines.append(f"{i}. Waited {action.duration_ms}ms")
            elif isinstance(action, RequestUpdatedImageAction):
                lines.append(f"{i}. Requested updated image")
        return "\n".join(lines)
    
    def solve_loop(
        self,
        get_image: Callable[[], str],
        context: str = "",
        max_steps: int = 10,
        end_condition: Optional[Callable[[], bool]] = None,
    ) -> Iterator[CaptchaAction]:
        """
        Solve a captcha with automatic looping.
        
        This generator yields actions one at a time until:
        - max_steps is reached
        - end_condition() returns True
        - The planner returns "done" action
        
        Args:
            get_image: Callable that returns the current image path
                       (called before each step to get fresh screenshot)
            context: Context string for the captcha
            max_steps: Maximum number of steps before giving up
            end_condition: Optional callable that returns True when captcha is solved
        
        Yields:
            CaptchaAction objects to execute
        
        Example:
            def get_screenshot():
                return take_screenshot("captcha_area.png")
            
            def is_solved():
                return not captcha_element.is_visible()
            
            for action in solver.solve_loop(get_screenshot, "Solve captcha", max_steps=10, end_condition=is_solved):
                if isinstance(action, ClickAction):
                    pyautogui.click(action.coordinates[0], action.coordinates[1])
                elif isinstance(action, WaitAction):
                    time.sleep(action.duration_ms / 1000)
                # ... handle other actions
        """
        # Reset history for new solve session
        self._action_history = []
        
        for step in range(max_steps):
            print(f"\n{'='*50}")
            print(f"Step {step + 1}/{max_steps}")
            print(f"{'='*50}")
            
            # Check end condition first
            if end_condition is not None and end_condition():
                print("End condition met - captcha solved!")
                return
            
            # Get current image
            image_path = get_image()
            
            # Get next action
            action = self.solve_step(image_path, context, include_history=True)
            
            # Check for "done" signal (wait with 0 duration)
            if isinstance(action, WaitAction) and action.duration_ms == 0:
                print("Planner indicates captcha is done")
                return
            
            # Yield the action for execution
            yield action
            
            # For request_updated_image, we just continue to next iteration
            # The caller should update the image via get_image()
        
        print(f"Reached max steps ({max_steps}) without solving")
    
    def reset_history(self):
        """Clear the action history."""
        self._action_history = []
    
    def visualize_attention(
        self,
        image_path: str,
        target_description: str,
        output_path: Optional[str] = None
    ):
        """
        Visualize where the attention extractor focuses for a given target.
        
        Useful for debugging and understanding the model's behavior.
        
        Args:
            image_path: Path to the image
            target_description: What to focus on
            output_path: Where to save the visualization (optional)
        
        Returns:
            numpy array of the visualization
        """
        return self.attention_extractor.visualize_attention(
            image_path,
            target_description,
            output_path
        )


# Convenience function for simple usage
def solve_captcha(
    image_path: str,
    context: str = "Solve this captcha",
    **kwargs
) -> CaptchaAction:
    """
    Convenience function to solve a single step of a captcha.
    
    Args:
        image_path: Path to the captcha image
        context: Context/instructions for solving
        **kwargs: Additional arguments passed to CaptchaSolver
    
    Returns:
        The next CaptchaAction to perform
    """
    solver = CaptchaSolver(**kwargs)
    return solver.solve_step(image_path, context)
