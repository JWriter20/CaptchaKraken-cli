"""
CaptchaSolver - Main entry point for solving captchas.

This module orchestrates the intelligent two-stage solving process:
1. ActionPlanner - Determines what action to take with reasoning
2. AttentionExtractor - Finds where to perform the action using detection/focus

Enhanced with:
- Self-questioning and iterative refinement
- Detection-based element mapping
- Support for numbered overlay elements
- Multi-click operations

Usage:
    # Ollama
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
    
    # Single step
    action = solver.solve_step("captcha.png")
    
    # Full loop
    for action in solver.solve_loop(get_image, max_steps=10, end_condition=is_done):
        execute_action(action)
"""

import os
import sys
from typing import List, Optional, Callable, Iterator, Union, Dict, Any
from pathlib import Path
from PIL import Image

from .action_types import (
    CaptchaAction,
    ClickAction,
    DragAction,
    TypeAction,
    WaitAction,
    RequestUpdatedImageAction,
    VerifyAction,
)
from .planner import ActionPlanner, PlannedAction
from .attention import AttentionExtractor
from .overlay import add_drag_overlay


class CaptchaSolver:
    """
    Intelligent two-stage captcha solver using LLM planning + attention-based coordinate extraction.
    
    Stage 1 (ActionPlanner): Ask an LLM (via Ollama or API provider) what action is needed
    Stage 2 (AttentionExtractor): Use Moondream for detection/focus to find precise coordinates
    
    Enhanced features:
    - Detection-based element finding
    - Self-questioning for refinement
    - Multi-click support
    - Numbered element mapping
    
    Args:
        model: Model name for planning (default: "hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M" for ollama)
        provider: API provider ("ollama", "openai", "gemini", or "deepseek"). Default: "ollama"
        api_key: API key (required for non-ollama providers)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: str = "ollama",
        api_key: Optional[str] = None,
    ):
        # Set default model based on provider
        if model is None:
            if provider == "ollama":
                model = "hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M"
            elif provider == "openai":
                model = "gpt-4o"
            elif provider == "gemini":
                model = "gemini-2.0-flash-exp"
            elif provider == "deepseek":
                model = "deepseek-chat"
            else:
                raise ValueError(f"Unknown provider '{provider}' and no model specified")
        
        # Validate required parameters
        if provider != "ollama" and not api_key:
            raise ValueError(f"api_key is required for provider '{provider}'")
        
        # Stage 1: Planner setup based on provider
        planner_kwargs = {
            "backend": provider,
            "model": model,
        }
        
        if provider == "openai":
            planner_kwargs["openai_api_key"] = api_key
        elif provider == "gemini":
            planner_kwargs["gemini_api_key"] = api_key
        elif provider == "deepseek":
            planner_kwargs["openai_api_key"] = api_key
            planner_kwargs["openai_base_url"] = "https://api.deepseek.com/v1"
        
        self.planner = ActionPlanner(**planner_kwargs)
        
        # Stage 2: Attention extractor always uses Moondream
        self.attention_extractor = AttentionExtractor(
            model="vikhyatk/moondream2",
            device=None,  # Auto-detect
            backend="moondream",
        )
        
        # Track action history for context
        self._action_history: List[CaptchaAction] = []
        
        # Cache for image information
        self._current_image_size: Optional[tuple] = None
    
    def solve_step(
        self,
        image_path: str,
        context: str = "",
    ) -> CaptchaAction:
        """
        Solve a single step of the captcha using the specialized flow.
        This flow:
        1) Classify captcha into checkbox | split-image | images | drag_puzzle | text
        2) Use a dedicated handler for that captcha type
        3) Fall back to the legacy planner if classification or specialized handling fails
        """
        # Resolve path
        image_path = str(Path(image_path).resolve())
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Get image dimensions
        with Image.open(image_path) as img:
            self._current_image_size = img.size
        
        # Step 1: classify captcha type
        classification = self.planner.classify_captcha(image_path, context)
        captcha_kind = (classification.get("captcha_kind") or "unknown").lower()
        drag_variant = (classification.get("drag_variant") or "unknown").lower()
        print(f"[Specialized] Classified as: {captcha_kind} (drag_variant={drag_variant})", file=sys.stderr)
        
        action: Optional[CaptchaAction] = None
        
        try:
            if captcha_kind == "checkbox":
                action = self._solve_checkbox(image_path, context)
            elif captcha_kind in ("split-image", "images"):
                action = self._solve_image_selection(
                    image_path,
                    context,
                    captcha_kind=captcha_kind,
                )
            elif captcha_kind == "drag_puzzle":
                action = self._solve_drag_puzzle(
                    image_path,
                    context,
                    drag_variant=drag_variant,
                )
            elif captcha_kind == "text":
                action = self._solve_text_captcha(image_path)
        except Exception as e:
            print(f"[Specialized] Handler error: {e}", file=sys.stderr)
            action = None
        
        # Fallback to legacy planner if needed
        if action is None:
            print("[Fallback] Falling back to legacy planner flow", file=sys.stderr)
            action = self._solve_with_legacy_planner(image_path, context)
        
        # Track history
        self._action_history.append(action)
        return action
    
    
    # ------------------------------------------------------------------
    # Specialized handlers (AlgoImprovements-aligned)
    # ------------------------------------------------------------------
    def _solve_checkbox(self, image_path: str, prompt_text: str) -> ClickAction:
        """Handle simple checkbox captchas with a precise focus call."""
        target_desc = prompt_text or "the checkbox"
        x_pct, y_pct = self.attention_extractor.focus(image_path, target_desc)
        x_px, y_px = self._pct_to_pixels(x_pct, y_pct)
        
        return ClickAction(
            action="click",
            coordinates=[x_px, y_px],
            point_percent=[x_pct, y_pct],
        )
    
    def _solve_image_selection(
        self,
        image_path: str,
        prompt_text: str,
        captcha_kind: str,
    ) -> ClickAction:
        """Handle split-image or images captchas via detection."""
        target_info = self.planner.plan_detection_target(
            image_path, prompt_text, captcha_kind=captcha_kind
        )
        target_label = (target_info.get("target_to_detect") or prompt_text or "target object").strip()
        print(f"[Specialized] Detecting target: {target_label}")
        
        detections = self.attention_extractor.detect_objects(image_path, target_label)
        
        bounding_boxes_pct: List[List[float]] = []
        bounding_boxes_px: List[List[float]] = []
        all_coordinates: List[List[float]] = []
        all_coordinates_pct: List[List[float]] = []
        
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            bbox_pct = self._ensure_bbox_normalized(bbox)
            bounding_boxes_pct.append(bbox_pct)
            bounding_boxes_px.append(self._bbox_pct_to_px(bbox_pct))
            center_pct = self._bbox_center_pct(bbox_pct)
            all_coordinates.append(list(self._pct_to_pixels_tuple(center_pct)))
            all_coordinates_pct.append(list(center_pct))
        
        if not bounding_boxes_pct:
            raise ValueError(f"Detection returned no boxes for target '{target_label}'")
        
        print(f"[Specialized] Found {len(bounding_boxes_pct)} detections")
        
        return ClickAction(
            action="click",
            bounding_boxes=bounding_boxes_pct,
            bounding_boxes_px=bounding_boxes_px,
            all_coordinates=all_coordinates if len(all_coordinates) > 1 else None,
            all_coordinates_pct=all_coordinates_pct if len(all_coordinates_pct) > 1 else None,
            coordinates=all_coordinates[0] if all_coordinates else None,
            point_percent=all_coordinates_pct[0] if all_coordinates_pct else None,
        )
    
    def _solve_drag_puzzle(
        self,
        image_path: str,
        prompt_text: str,
        drag_variant: str = "unknown",
    ) -> DragAction:
        """Handle drag puzzles with iterative or logical flows."""
        strategy = self.planner.plan_drag_strategy(image_path, prompt_text)
        variant = (strategy.get("drag_type") or drag_variant or "logical").lower()
        print(f"[Specialized] Drag strategy variant: {variant}")
        
        if variant == "template_matching":
            # Now uses the iterative visual approach
            return self._solve_iterative_drag(image_path, strategy, prompt_text)
        return self._solve_logical_drag(image_path, strategy)
    
    def _solve_logical_drag(self, image_path: str, strategy: Dict[str, Any]) -> DragAction:
        """Logical drag: use detection/focus for draggable and destination."""
        draggable_prompt = strategy.get("draggable_prompt") or "movable piece"
        destination_prompt = strategy.get("destination_prompt") or "target location"
        
        # 1. Find draggable
        source_bbox = self._detect_bbox_with_fallback(image_path, draggable_prompt)
        
        # 2. Find destination (bee logic: finding destination and reasoning)
        target_bbox = self._detect_bbox_with_fallback(image_path, destination_prompt)
        
        source_center_pct = self._bbox_center_pct(source_bbox)
        target_center_pct = self._bbox_center_pct(target_bbox)
        
        source_px = self._pct_to_pixels_tuple(source_center_pct)
        target_px = self._pct_to_pixels_tuple(target_center_pct)
        
        return DragAction(
            action="drag",
            source_coordinates=list(source_px),
            source_coordinates_pct=list(source_center_pct),
            target_coordinates=list(target_px),
            target_coordinates_pct=list(target_center_pct),
        )
    
    def _solve_iterative_drag(self, image_path: str, strategy: Dict[str, Any], prompt_text: str) -> DragAction:
        """
        Iterative visual solver for drag puzzles.
        1. Locate draggable piece
        2. Visual loop: Propose move -> Visual Feedback -> Refine -> Repeat
        """
        draggable_prompt = strategy.get("draggable_prompt") or "movable piece"
        
        # 1. Locate the draggable piece
        print(f"[Iterative] Locating draggable: {draggable_prompt}")
        source_bbox_pct = self._detect_bbox_with_fallback(image_path, draggable_prompt)
        if not source_bbox_pct:
            raise ValueError("Could not locate draggable piece for iterative solve")
            
        source_bbox_px = self._bbox_pct_to_px(source_bbox_pct)
        source_center_pct = self._bbox_center_pct(source_bbox_pct)
        
        # Working copy for overlays
        import shutil
        import tempfile
        
        # Create a temp file that preserves the extension
        ext = os.path.splitext(image_path)[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
            work_path = tf.name
        shutil.copy2(image_path, work_path)
        
        try:
            # 2. Initial Proposal
            # Draw just the source border first so the model knows what we are moving, plus grid
            add_drag_overlay(work_path, source_bbox_px)
            
            # Ask where to go (pass bbox for extra context)
            initial_plan = self.planner.get_drag_destination(
                work_path,
                prompt_text or "",
                draggable_bbox_pct=source_bbox_pct,
            )
            
            target_center_pct = [
                initial_plan.get("target_x", source_center_pct[0]),
                initial_plan.get("target_y", source_center_pct[1]),
            ]
                
            # Clamp to [0,1]
            target_center_pct = [
                max(0.0, min(1.0, target_center_pct[0])),
                max(0.0, min(1.0, target_center_pct[1]))
            ]
            
            # 3. Refinement Loop
            max_iterations = 3
            current_target_pct = target_center_pct
            
            for i in range(max_iterations):
                print(f"[Iterative] Iteration {i+1}: Proposed target {current_target_pct}")
                
                # Reset working image from original to clear previous arrows
                shutil.copy2(image_path, work_path)
                
                # Calculate target bbox (same size as source)
                w_pct = source_bbox_pct[2] - source_bbox_pct[0]
                h_pct = source_bbox_pct[3] - source_bbox_pct[1]
                
                target_bbox_pct = [
                    current_target_pct[0] - w_pct/2,
                    current_target_pct[1] - h_pct/2,
                    current_target_pct[0] + w_pct/2,
                    current_target_pct[1] + h_pct/2
                ]
                
                # Draw full overlay: Source Border + Arrow + Target Box + Grid
                target_bbox_px = self._bbox_pct_to_px(target_bbox_pct)
                target_center_px = self._pct_to_pixels_tuple(current_target_pct)
                
                add_drag_overlay(
                    work_path, 
                    source_bbox_px, 
                    target_bbox=target_bbox_px, 
                    target_center=target_center_px,
                )
                
                # Ask for refinement
                refinement = self.planner.refine_drag(work_path)
                print(f"[Iterative] Feedback: {refinement.get('status')} - {refinement.get('adjustment')}")
                
                if refinement.get("status") == "correct":
                    print("[Iterative] Model accepted solution")
                    break
                
                # Apply adjustment
                adj = refinement.get("adjustment", {})
                dx = adj.get("x_offset", 0) or 0
                dy = adj.get("y_offset", 0) or 0
                
                # If adjustment is tiny, assume done
                if abs(dx) < 0.01 and abs(dy) < 0.01:
                    print("[Iterative] Adjustment too small, stopping")
                    break
                    
                current_target_pct[0] += dx
                current_target_pct[1] += dy
                
                # Clamp
                current_target_pct = [
                    max(0.0, min(1.0, current_target_pct[0])),
                    max(0.0, min(1.0, current_target_pct[1]))
                ]
            
            # Final result
            target_px = self._pct_to_pixels_tuple(current_target_pct)
            source_px = self._pct_to_pixels_tuple(source_center_pct)
            
            return DragAction(
                action="drag",
                source_coordinates=list(source_px),
                source_coordinates_pct=list(source_center_pct),
                target_coordinates=list(target_px),
                target_coordinates_pct=list(current_target_pct),
            )
            
        finally:
            if os.path.exists(work_path):
                os.unlink(work_path)

    def _solve_text_captcha(self, image_path: str) -> TypeAction:
        """Handle warped-text captchas via LLM text read."""
        text = self.planner.read_text(image_path)
        return TypeAction(action="type", text=text or "")
    
    def _solve_with_legacy_planner(
        self,
        image_path: str,
        context: str,
    ) -> CaptchaAction:
        """Preserve the previous two-stage planner flow as a fallback."""
        full_context = context
        if self._action_history:
            history_str = self._format_history()
            full_context = f"{context}\n\nPrevious actions taken:\n{history_str}"
        
        planned = self.planner.plan(image_path, full_context)
        return self._create_action(image_path, planned)
    
    # ------------------------------------------------------------------
    # Geometry and detection helpers
    # ------------------------------------------------------------------
    def _detect_bbox_with_fallback(
        self,
        image_path: str,
        prompt: str,
        allow_empty: bool = False,
    ) -> Optional[List[float]]:
        """Detect a bounding box via detect(); fall back to focus -> box."""
        detections = self.attention_extractor.detect_objects(image_path, prompt)
        if detections:
            bbox = detections[0].get("bbox")
            if bbox and len(bbox) >= 4:
                return self._ensure_bbox_normalized(bbox)
        
        if allow_empty:
            return None
        
        x_pct, y_pct = self.attention_extractor.focus(image_path, prompt)
        return self._expand_point_to_bbox((x_pct, y_pct))
    
    def _expand_point_to_bbox(self, point_pct: tuple) -> List[float]:
        """Create a small bounding box around a normalized point."""
        x_pct, y_pct = point_pct
        half_size = 0.06  # 12% box side
        x1 = max(0.0, x_pct - half_size)
        y1 = max(0.0, y_pct - half_size)
        x2 = min(1.0, x_pct + half_size)
        y2 = min(1.0, y_pct + half_size)
        return [x1, y1, x2, y2]
    
    def _pct_to_pixels(self, x_pct: float, y_pct: float) -> tuple:
        """Convert normalized point to pixel coordinates if size known."""
        if not self._current_image_size:
            return x_pct, y_pct
        return (
            x_pct * self._current_image_size[0],
            y_pct * self._current_image_size[1],
        )
    
    def _pct_to_pixels_tuple(self, point_pct: tuple) -> tuple:
        return self._pct_to_pixels(point_pct[0], point_pct[1])
    
    def _pixels_to_pct(self, point_px: tuple) -> tuple:
        """Convert pixel coordinates to normalized percentages."""
        if not self._current_image_size:
            return point_px
        return (
            float(point_px[0]) / self._current_image_size[0],
            float(point_px[1]) / self._current_image_size[1],
        )
    
    def _bbox_pct_to_px(self, bbox_pct: List[float]) -> List[float]:
        """Convert normalized bbox to pixel coordinates."""
        if not self._current_image_size:
            return bbox_pct
        w, h = self._current_image_size
        x1, y1, x2, y2 = bbox_pct[:4]
        return [x1 * w, y1 * h, x2 * w, y2 * h]
    
    def _bbox_px_to_pct(self, bbox_px: tuple) -> List[float]:
        """Convert pixel bbox to normalized format."""
        if not self._current_image_size:
            return list(bbox_px)
        w, h = self._current_image_size
        x1, y1, x2, y2 = bbox_px
        return [x1 / w, y1 / h, x2 / w, y2 / h]
    
    def _bbox_center_pct(self, bbox_pct: List[float]) -> tuple:
        """Return center of normalized bbox."""
        x1, y1, x2, y2 = bbox_pct[:4]
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _ensure_bbox_normalized(self, bbox: List[float]) -> List[float]:
        """Normalize bbox to [0,1] corner format regardless of input scale."""
        if len(bbox) < 4:
            return bbox
        
        # If values already <= 1 assume normalized corner format
        if all(v <= 1.0 for v in bbox[:4]):
            return bbox[:4]
        
        # Otherwise assume pixel format [x1, y1, x2, y2] or [x, y, w, h]
        if self._current_image_size:
            img_w, img_h = self._current_image_size
            x1, y1, x2, y2 = bbox[:4]
            # If already likely in corner format, keep; otherwise treat x2/y2 as width/height
            if not (x2 > x1 and y2 > y1 and x2 <= img_w and y2 <= img_h):
                x2 = x1 + x2
                y2 = y1 + y2
            
            return [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]
        return bbox[:4]
    
    def _create_action(
        self,
        image_path: str,
        planned: PlannedAction,
    ) -> CaptchaAction:
        """
        Convert a PlannedAction to a CaptchaAction, extracting coordinates if needed.
        """
        if planned.action_type == "click":
            return self._create_click_action(image_path, planned)
        
        elif planned.action_type == "drag":
            return self._create_drag_action(image_path, planned)
        
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
        
        elif planned.action_type == "verify":
            return self._create_verify_action(image_path, planned)
        
        elif planned.action_type == "done":
            # Return a wait with 0 duration to signal completion
            return WaitAction(action="wait", duration_ms=0)
        
        else:
            raise ValueError(f"Unknown action type: {planned.action_type}")
    
    def _create_click_action(
        self,
        image_path: str,
        planned: PlannedAction,
    ) -> ClickAction:
        """Create a ClickAction with coordinates."""
        
        # Use attention-based extraction
        print(f"[Stage 2] Extracting click coordinates for: {planned.target_description}")
        x_pct, y_pct = self.attention_extractor.extract_coordinates(
            image_path,
            planned.target_description or "the target element"
        )
        
        # Convert to pixels if we have image size
        if self._current_image_size:
            x = x_pct * self._current_image_size[0]
            y = y_pct * self._current_image_size[1]
        else:
            x, y = x_pct, y_pct
        
        print(f"[Stage 2] Coordinates: ({x}, {y})")
        return ClickAction(
            action="click",
            coordinates=[x, y],
            point_percent=[x_pct, y_pct]
        )
    
    def _create_drag_action(
        self,
        image_path: str,
        planned: PlannedAction,
    ) -> DragAction:
        """Create a DragAction with source and target coordinates."""
        
        source_coords = None
        target_coords = None
        
        # Extract source using attention
        print(f"[Stage 2] Extracting drag source: {planned.target_description}")
        
        # First try detect() for more accurate results
        if planned.target_description:
            detections = self.attention_extractor.detect_objects(
                image_path,
                planned.target_description
            )
            if detections:
                bbox = detections[0]['bbox']
                x_pct = (bbox[0] + bbox[2]) / 2
                y_pct = (bbox[1] + bbox[3]) / 2
            else:
                # Fall back to focus/point
                result = self.attention_extractor.focus(
                    image_path,
                    planned.target_description
                )
                x_pct, y_pct = result if result else (0.5, 0.5)
        else:
            x_pct, y_pct = 0.5, 0.5
        
        if self._current_image_size:
            source_coords = (x_pct * self._current_image_size[0], y_pct * self._current_image_size[1])
        else:
            source_coords = (x_pct, y_pct)
        
        # Extract target using attention
        print(f"[Stage 2] Extracting drag target: {planned.drag_target_description}")
        
        if planned.drag_target_description:
            detections = self.attention_extractor.detect_objects(
                image_path,
                planned.drag_target_description
            )
            if detections:
                bbox = detections[0]['bbox']
                x_pct = (bbox[0] + bbox[2]) / 2
                y_pct = (bbox[1] + bbox[3]) / 2
            else:
                result = self.attention_extractor.focus(
                    image_path,
                    planned.drag_target_description
                )
                x_pct, y_pct = result if result else (0.5, 0.5)
        else:
            x_pct, y_pct = 0.5, 0.5
        
        if self._current_image_size:
            target_coords = (x_pct * self._current_image_size[0], y_pct * self._current_image_size[1])
        else:
            target_coords = (x_pct, y_pct)
        
        print(f"[Stage 2] Source: {source_coords}, Target: {target_coords}")
        
        # Calculate percentages for response if not already available
        source_pct = None
        target_pct = None
        
        if self._current_image_size:
             w, h = self._current_image_size
             source_pct = [source_coords[0]/w, source_coords[1]/h]
             target_pct = [target_coords[0]/w, target_coords[1]/h]
        
        return DragAction(
            action="drag",
            source_coordinates=list(source_coords),
            source_coordinates_pct=source_pct,
            target_coordinates=list(target_coords),
            target_coordinates_pct=target_pct
        )
    
    def _create_verify_action(
        self,
        image_path: str,
        planned: PlannedAction,
    ) -> VerifyAction:
        """Create a VerifyAction."""
        return VerifyAction(action="verify")
    def _format_history(self) -> str:
        """Format action history for context."""
        lines = []
        for i, action in enumerate(self._action_history[-5:], 1):  # Last 5 actions
            if isinstance(action, ClickAction):
                if getattr(action, "bounding_boxes", None):
                    lines.append(f"{i}. Clicked {len(action.bounding_boxes)} detected boxes")
                else:
                    lines.append(f"{i}. Clicked at {action.coordinates}")
            elif isinstance(action, DragAction):
                lines.append(f"{i}. Dragged from {action.source_coordinates} to {action.target_coordinates}")
            elif isinstance(action, TypeAction):
                lines.append(f"{i}. Typed: '{action.text}'")
            elif isinstance(action, WaitAction):
                lines.append(f"{i}. Waited {action.duration_ms}ms")
            elif isinstance(action, RequestUpdatedImageAction):
                lines.append(f"{i}. Requested updated image")
            elif isinstance(action, VerifyAction):
                lines.append(f"{i}. Clicked verify button")
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
            context: Context string for the captcha
            max_steps: Maximum number of steps before giving up
            end_condition: Optional callable that returns True when captcha is solved
        
        Yields:
            CaptchaAction objects to execute
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
            action = self.solve_step(image_path, context)
            
            # Check for "done" signal (wait with 0 duration)
            if isinstance(action, WaitAction) and action.duration_ms == 0:
                print("Planner indicates captcha is done")
                return
            
            # Yield the action for execution
            yield action
        
        print(f"Reached max steps ({max_steps}) without solving")
    
    def solve_with_refinement(
        self,
        image_path: str,
        context: str = "",
        max_questions: int = 3,
    ) -> CaptchaAction:
        """
        Solve with self-questioning refinement.
        
        This method:
        1. Gets initial plan
        2. Asks clarifying questions about uncertain elements
        3. Refines the plan based on answers
        4. Returns the refined action
        
        Args:
            image_path: Path to the captcha image
            context: Additional context
            max_questions: Maximum number of clarifying questions to ask
        
        Returns:
            Refined CaptchaAction
        """
        image_path = str(Path(image_path).resolve())
        
        # Get initial plan
        planned = self.planner.plan(image_path, context)
        
        print(f"[Initial Plan] {planned.action_type}")
        print(f"[Initial Plan] Reasoning: {planned.reasoning}")
        
        # Create and return action
        if self._current_image_size is None:
            with Image.open(image_path) as img:
                self._current_image_size = img.size
        
        action = self._create_action(image_path, planned)
        self._action_history.append(action)
        
        return action
    
    def reset_history(self):
        """Clear the action history."""
        self._action_history = []
        self._current_image_size = None
    
    def visualize_attention(
        self,
        image_path: str,
        target_description: str,
        output_path: Optional[str] = None
    ):
        """
        Visualize where the attention extractor focuses for a given target.
        
        Useful for debugging and understanding the model's behavior.
        """
        return self.attention_extractor.visualize_attention(
            image_path,
            target_description,
            output_path or "attention_visualization.png"
        )
    
    def visualize_detections(
        self,
        image_path: str,
        object_class: str,
        output_path: Optional[str] = None,
    ):
        """
        Visualize object detections.
        """
        detections = self.attention_extractor.detect_objects(image_path, object_class)
        
        return self.attention_extractor.visualize_detections(
            image_path,
            detections,
            output_path or "detections_visualization.png"
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
