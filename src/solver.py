"""
CaptchaSolver - Simplified captcha solver using LLM planning + vision tools.

Flow:
1. Grid detection - if grid found, use numbered overlay and ask planner to select
2. General solving - let planner use detect/point tools to find targets
3. Drag simulation - iterative refinement with visual feedback

Usage:
    from src import CaptchaSolver

    solver = CaptchaSolver(provider="gemini", api_key="...")
    actions = solver.solve("captcha.png", "Select all traffic lights")
"""

import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Debug flag - set via CAPTCHA_DEBUG=1 environment variable
DEBUG = os.getenv("CAPTCHA_DEBUG", "0") == "1"


def _debug_log(message: str) -> None:
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[Solver DEBUG] {message}", file=sys.stderr)

from PIL import Image

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


class CaptchaSolver:
    """
    Simplified captcha solver.

    Supports:
    - Grid selection captchas (reCAPTCHA, hCaptcha image grids)
    - Checkbox captchas
    - Drag puzzles with iterative refinement
    - Text captchas
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: str = "gemini",
        api_key: Optional[str] = None,
    ):
        # Restrict providers to the supported set
        if provider not in {"ollama", "gemini"}:
            raise ValueError(f"Unsupported provider '{provider}'. Supported providers are: 'ollama', 'gemini'.")

        # Set default model based on provider
        if model is None:
            if provider == "ollama":
                model = "llama3.2-vision"
            elif provider == "gemini":
                model = "gemini-2.5-flash-lite"

        # Validate required parameters
        if provider != "ollama" and not api_key:
            raise ValueError(f"api_key is required for provider '{provider}'")

        # Setup planner
        planner_kwargs: dict = {
            "backend": provider,
            "model": model,
        }

        if provider == "gemini" and api_key:
            planner_kwargs["gemini_api_key"] = api_key

        self.planner = ActionPlanner(**planner_kwargs)  # type: ignore[arg-type]

        # Setup attention extractor (Moondream for detect/point)
        self.attention = AttentionExtractor(
            model="vikhyatk/moondream2",
            device=None,  # Auto-detect
        )

        # Image size cache
        self._image_size: Optional[Tuple[int, int]] = None

    def solve(
        self,
        image_path: str,
        instruction: str = "",
    ) -> Union[CaptchaAction, List[ClickAction]]:
        """
        Solve a captcha image.

        Args:
            image_path: Path to the captcha image
            instruction: Optional instruction text (e.g., "Select all traffic lights")

        Returns:
            CaptchaAction or List[ClickAction] for grid selections
        """
        # Resolve path and get image size
        image_path = str(Path(image_path).resolve())
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            self._image_size = img.size

        # 1. Check for grid structure
        grid_boxes = get_grid_bounding_boxes(image_path)
        if grid_boxes:
            print(f"[Solver] Detected grid with {len(grid_boxes)} cells", file=sys.stderr)
            return self._solve_grid(image_path, instruction, grid_boxes)

        # 2. General solving with tool use
        print("[Solver] Using general solving flow", file=sys.stderr)
        return self._solve_general(image_path, instruction)

    def _solve_grid(
        self,
        image_path: str,
        instruction: str,
        grid_boxes: List[Tuple[int, int, int, int]],
    ) -> List[ClickAction]:
        """
        Solve grid selection captcha.

        1. Apply numbered overlay
        2. Ask planner which squares to select
        3. Return ClickActions for selected squares
        """
        n = len(grid_boxes)
        if n == 9:
            rows, cols = 3, 3
        elif n == 16:
            rows, cols = 4, 4
        else:
            cols = int(math.sqrt(n))
            rows = math.ceil(n / cols)

        _debug_log(f"_solve_grid called with {n} cells ({rows}x{cols})")
        _debug_log(f"Instruction passed to planner: '{instruction}'")

        # Create temp image with numbered overlay
        ext = os.path.splitext(image_path)[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
            overlay_path = tf.name

        try:
            # Prepare overlays - bbox in [x, y, w, h] pixel format for overlay function
            overlays = []
            for i, (x1, y1, x2, y2) in enumerate(grid_boxes):
                w = x2 - x1
                h = y2 - y1
                overlays.append({"bbox": [x1, y1, w, h], "number": i + 1, "color": "#E74C3C"})

            add_overlays_to_image(image_path, overlays, output_path=overlay_path)
            _debug_log(f"Generated overlay image at: {overlay_path}")

            # Ask planner which squares to select
            selected_numbers = self.planner.get_grid_selection(instruction, overlay_path, rows=rows, cols=cols)

            print(f"[Solver] Grid selection: {selected_numbers}", file=sys.stderr)

            # Convert to ClickActions
            actions = []
            assert self._image_size is not None
            img_w, img_h = self._image_size
            for num in selected_numbers:
                idx = num - 1
                if 0 <= idx < len(grid_boxes):
                    x1, y1, x2, y2 = grid_boxes[idx]

                    # Convert to normalized coordinates
                    bbox_pct = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]

                    actions.append(
                        ClickAction(
                            action="click",
                            target_number=num,
                            target_bounding_box=bbox_pct,
                        )
                    )

            return actions

        finally:
            if os.path.exists(overlay_path):
                os.unlink(overlay_path)

    def _solve_general(
        self,
        image_path: str,
        instruction: str,
    ) -> CaptchaAction:
        """
        General solving using planner with detect/point tools.

        The planner can request tool calls (detect, point) or return a direct action.
        """
        max_tool_calls = 3

        for _ in range(max_tool_calls):
            result = self.planner.plan_with_tools(image_path, instruction)

            # Handle tool calls
            if result.get("tool_call"):
                tool = result["tool_call"]["name"]
                args = result["tool_call"]["args"]

                if tool == "detect":
                    object_class = args.get("object_class", "target")
                    print(f"[Solver] Tool call: detect('{object_class}')", file=sys.stderr)
                    detections = self.attention.detect_objects(image_path, object_class)

                    if detections:
                        bbox = detections[0]["bbox"]
                        return ClickAction(
                            action="click",
                            target_bounding_box=bbox,
                        )
                    else:
                        # No detections - try point as fallback
                        print("[Solver] No detections, falling back to point()", file=sys.stderr)
                        x, y = self.attention.extract_coordinates(image_path, object_class)
                        return ClickAction(
                            action="click",
                            target_coordinates=[x, y],
                        )

                elif tool == "point":
                    target = args.get("target", "target")
                    print(f"[Solver] Tool call: point('{target}')", file=sys.stderr)
                    x, y = self.attention.extract_coordinates(image_path, target)
                    return ClickAction(
                        action="click",
                        target_coordinates=[x, y],
                    )

                continue  # Try again if tool call didn't resolve

            # Handle direct actions
            action_type = result.get("action_type", "click")

            if action_type == "type":
                return TypeAction(
                    action="type",
                    text=result.get("text", ""),
                )

            elif action_type == "wait":
                return WaitAction(
                    action="wait",
                    duration_ms=result.get("duration_ms", 500),
                )

            elif action_type == "drag":
                return self._solve_drag(
                    image_path,
                    instruction,
                    result.get("source_description"),
                    result.get("target_description"),
                )

            elif action_type == "click":
                target = result.get("target_description", "target")
                x, y = self.attention.extract_coordinates(image_path, target)
                return ClickAction(
                    action="click",
                    target_coordinates=[x, y],
                )

            elif action_type == "done":
                return WaitAction(action="wait", duration_ms=0)

        # Fallback after max tool calls
        return WaitAction(action="wait", duration_ms=0)

    def _solve_drag(
        self,
        image_path: str,
        instruction: str,
        source_description: Optional[str],
        target_description: Optional[str],
        max_iterations: int = 5,
    ) -> DragAction:
        """
        Solve drag puzzle with iterative refinement.

        Uses visual feedback loop:
        1. Locate source element
        2. Initial target estimate from point() or description
        3. Draw overlay (arrow + destination box)
        4. Ask model for adjustment (relative, e.g., "+5%, -2%")
        5. Repeat until satisfied or max iterations reached

        The model returns relative adjustments like:
        - dx: +0.05 (5% to the right)
        - dy: -0.02 (2% up)
        """
        source_desc = source_description or "draggable piece"

        # 1. Find source
        print(f"[Solver] Finding drag source: '{source_desc}'", file=sys.stderr)
        source_x, source_y = self.attention.extract_coordinates(image_path, source_desc)
        source_coords = [source_x, source_y]

        print(f"[Solver] Drag source at: ({source_x:.2%}, {source_y:.2%})", file=sys.stderr)

        # 2. Initial target estimate
        if target_description:
            print(f"[Solver] Finding drag target: '{target_description}'", file=sys.stderr)
            target_x, target_y = self.attention.extract_coordinates(image_path, target_description)
        else:
            # Default: slightly to the right of source
            target_x, target_y = min(source_x + 0.2, 0.9), source_y

        current_target = [target_x, target_y]
        print(f"[Solver] Initial target estimate: ({target_x:.2%}, {target_y:.2%})", file=sys.stderr)

        # 3. Iterative refinement with visual feedback
        history: List[dict] = []

        ext = os.path.splitext(image_path)[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
            work_path = tf.name

        assert self._image_size is not None
        img_w, img_h = self._image_size

        try:
            for i in range(max_iterations):
                # Reset working image
                shutil.copy2(image_path, work_path)

                # Convert to pixels for overlay
                w, h = img_w, img_h
                source_px = [source_x * w, source_y * h]
                target_px = [current_target[0] * w, current_target[1] * h]

                # Source bbox (small box around point)
                box_size = 0.05
                source_bbox_px = [
                    source_px[0] - box_size * w,
                    source_px[1] - box_size * h,
                    source_px[0] + box_size * w,
                    source_px[1] + box_size * h,
                ]

                # Target bbox (same size)
                target_bbox_px = [
                    target_px[0] - box_size * w,
                    target_px[1] - box_size * h,
                    target_px[0] + box_size * w,
                    target_px[1] + box_size * h,
                ]

                # Draw overlay
                add_drag_overlay(
                    work_path,
                    source_bbox_px,
                    target_bbox=target_bbox_px,
                    target_center=tuple(target_px),
                )

                # Ask for refinement with history
                result = self.planner.refine_drag(
                    work_path,
                    instruction,
                    current_target,
                    history,
                )

                conclusion = result.get("conclusion", "")
                decision = result.get("decision", "accept")
                dx = result.get("dx", 0)
                dy = result.get("dy", 0)

                print(
                    f"[Solver] Iteration {i + 1}: '{conclusion}' -> {decision} (dx={dx:+.1%}, dy={dy:+.1%})",
                    file=sys.stderr,
                )

                # Record history
                history.append(
                    {
                        "destination": current_target.copy(),
                        "conclusion": conclusion,
                        "decision": decision,
                    }
                )

                # Check if done
                if decision == "accept" or (abs(dx) < 0.005 and abs(dy) < 0.005):
                    print("[Solver] Drag target accepted", file=sys.stderr)
                    break

                # Apply adjustment (relative)
                current_target[0] = max(0.0, min(1.0, current_target[0] + dx))
                current_target[1] = max(0.0, min(1.0, current_target[1] + dy))

        finally:
            if os.path.exists(work_path):
                os.unlink(work_path)

        return DragAction(
            action="drag",
            source_coordinates=source_coords,
            target_coordinates=current_target,
        )


# Convenience function
def solve_captcha(image_path: str, instruction: str = "", **kwargs) -> Union[CaptchaAction, List[ClickAction]]:
    """
    Convenience function to solve a captcha.

    Args:
        image_path: Path to the captcha image
        instruction: Instruction text for solving
        **kwargs: Additional arguments passed to CaptchaSolver

    Returns:
        CaptchaAction or List[ClickAction] for grid selections
    """
    solver = CaptchaSolver(**kwargs)
    return solver.solve(image_path, instruction)
