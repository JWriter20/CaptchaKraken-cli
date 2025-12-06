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
    solver = CaptchaSolver()
    
    # Single step
    action = solver.solve_step("captcha.png", "Solve this captcha")
    
    # Full loop with context
    for action in solver.solve_loop(get_image, context, max_steps=10, end_condition=is_done):
        execute_action(action)
    
    # Intelligent solve with element info
    action = solver.solve_step_intelligent(
        "captcha.png",
        context="Select all traffic lights",
        elements=[{"element_id": 1, "bbox": [0, 0, 100, 100]}, ...],
        prompt_text="Select all images with traffic lights"
    )
"""

import os
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
    CaptchaContext,
    InteractableElement,
)
# Template-matching utilities live at top-level of core src
from template_matching import (
    extract_segment,
    remove_background,
    match_template_with_mask,
    find_draggable_candidates,
)
from .planner import ActionPlanner, PlannedAction
from .attention import AttentionExtractor


class CaptchaSolver:
    """
    Intelligent two-stage captcha solver using LLM planning + attention-based coordinate extraction.
    
    Stage 1 (ActionPlanner): Ask an LLM (via Ollama or OpenAI API) what action is needed
    Stage 2 (AttentionExtractor): Use detection/focus to find precise coordinates
    
    Enhanced features:
    - Detection-based element finding
    - Self-questioning for refinement
    - Multi-click support
    - Numbered element mapping
    
    Args:
        planner_backend: Backend for action planning ("ollama" or "openai")
        planner_model: Model for planning (default: "hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M" for ollama, "gpt-4o" for openai)
        attention_model: HuggingFace model for attention extraction (default: "vikhyatk/moondream2")
        attention_backend: Backend for attention ("moondream", "qwen-vl", "florence")
        openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        openai_base_url: Custom OpenAI base URL (or set OPENAI_BASE_URL env var)
        device: Device for attention model ("cuda", "cpu", "mps"). Auto-detected if None.
    """
    
    def __init__(
        self,
        planner_backend: str = "ollama",
        planner_model: Optional[str] = None,
        attention_model: Optional[str] = None,
        attention_backend: str = "moondream",
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
            backend=attention_backend,
        )
        
        # Track action history for context
        self._action_history: List[CaptchaAction] = []
        
        # Cache for element information
        self._current_elements: Optional[List[dict]] = None
        self._current_image_size: Optional[tuple] = None
    
    def solve_step(
        self,
        image_path: str,
        context: str = "",
        include_history: bool = True,
        elements: Optional[List[dict]] = None,
        prompt_text: Optional[str] = None,
    ) -> CaptchaAction:
        """
        Solve a single step of the captcha using the specialized flow from
        AlgoImprovements.txt. This flow is:
        1) Classify captcha into checkbox | split-image | images | drag_puzzle | text
        2) Use a dedicated handler for that captcha type
        3) Fall back to the legacy planner if classification or specialized handling fails
        """
        # Resolve path
        image_path = str(Path(image_path).resolve())
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Store elements for coordinate mapping
        self._current_elements = elements
        
        # Get image dimensions
        with Image.open(image_path) as img:
            self._current_image_size = img.size
        
        # Step 1: classify captcha type
        classification = self.planner.classify_captcha(image_path, prompt_text or context)
        captcha_kind = (classification.get("captcha_kind") or "unknown").lower()
        drag_variant = (classification.get("drag_variant") or "unknown").lower()
        print(f"[Specialized] Classified as: {captcha_kind} (drag_variant={drag_variant})")
        
        action: Optional[CaptchaAction] = None
        
        try:
            if captcha_kind == "checkbox":
                action = self._solve_checkbox(image_path, prompt_text or context)
            elif captcha_kind in ("split-image", "images"):
                action = self._solve_image_selection(
                    image_path,
                    prompt_text or context,
                    captcha_kind=captcha_kind,
                )
            elif captcha_kind == "drag_puzzle":
                action = self._solve_drag_puzzle(
                    image_path,
                    prompt_text or context,
                    drag_variant=drag_variant,
                )
            elif captcha_kind == "text":
                action = self._solve_text_captcha(image_path)
        except Exception as e:
            print(f"[Specialized] Handler error: {e}")
            action = None
        
        # Fallback to legacy planner if needed
        if action is None:
            print("[Fallback] Falling back to legacy planner flow")
            action = self._solve_with_legacy_planner(
                image_path=image_path,
                context=context,
                include_history=include_history,
                elements=elements,
                prompt_text=prompt_text,
            )
        
        # Track history
        self._action_history.append(action)
        return action
    
    def solve_step_intelligent(
        self,
        image_path: str,
        context: str = "",
        elements: Optional[List[dict]] = None,
        prompt_text: Optional[str] = None,
        use_detection: bool = True,
    ) -> CaptchaAction:
        """
        Intelligent solve step; now simply routes through the specialized
        solve_step pipeline. Kept for backwards compatibility with callers.
        """
        return self.solve_step(
            image_path,
            context=context,
            include_history=True,
            elements=elements,
            prompt_text=prompt_text,
        )
    
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
        
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            bbox_pct = self._ensure_bbox_normalized(bbox)
            bounding_boxes_pct.append(bbox_pct)
            bounding_boxes_px.append(self._bbox_pct_to_px(bbox_pct))
            center_pct = self._bbox_center_pct(bbox_pct)
            all_coordinates.append(list(self._pct_to_pixels_tuple(center_pct)))
        
        if not bounding_boxes_pct:
            raise ValueError(f"Detection returned no boxes for target '{target_label}'")
        
        print(f"[Specialized] Found {len(bounding_boxes_pct)} detections")
        
        return ClickAction(
            action="click",
            bounding_boxes=bounding_boxes_pct,
            bounding_boxes_px=bounding_boxes_px,
            all_coordinates=all_coordinates if len(all_coordinates) > 1 else None,
            coordinates=all_coordinates[0] if all_coordinates else None,
        )
    
    def _solve_drag_puzzle(
        self,
        image_path: str,
        prompt_text: str,
        drag_variant: str = "unknown",
    ) -> DragAction:
        """Handle drag puzzles with template-matching or logical flows."""
        strategy = self.planner.plan_drag_strategy(image_path, prompt_text)
        variant = (strategy.get("drag_type") or drag_variant or "logical").lower()
        print(f"[Specialized] Drag strategy variant: {variant}")
        
        if variant == "template_matching":
            return self._solve_template_drag(image_path, strategy)
        return self._solve_logical_drag(image_path, strategy)
    
    def _solve_logical_drag(self, image_path: str, strategy: Dict[str, Any]) -> DragAction:
        """Logical drag: use detection/focus for draggable and destination."""
        draggable_prompt = strategy.get("draggable_prompt") or "movable piece"
        destination_prompt = strategy.get("destination_prompt") or "target location"
        
        source_bbox = self._detect_bbox_with_fallback(image_path, draggable_prompt)
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
    
    def _solve_template_drag(self, image_path: str, strategy: Dict[str, Any]) -> DragAction:
        """Template-matching drag using OpenCV helpers."""
        draggable_prompt = strategy.get("draggable_prompt") or "movable piece"
        
        # Try model-based detection first
        source_bbox_pct = self._detect_bbox_with_fallback(image_path, draggable_prompt, allow_empty=True)
        
        # If detection fails, fall back to heuristic candidates
        if source_bbox_pct is None:
            candidates = find_draggable_candidates(image_path)
            if candidates:
                source_bbox_px = candidates[0]["bbox"]
                source_bbox_pct = self._bbox_px_to_pct(source_bbox_px)
                print("[Specialized] Using heuristic draggable candidate")
        
        if source_bbox_pct is None:
            raise ValueError("Unable to locate draggable element for template drag")
        
        source_bbox_px = self._bbox_pct_to_px(source_bbox_pct)
        source_center_pct = self._bbox_center_pct(source_bbox_pct)
        source_center_px = self._pct_to_pixels_tuple(source_center_pct)
        
        # Extract and clean the draggable piece, then match
        segment = extract_segment(image_path, tuple(source_bbox_px))
        cleaned = remove_background(segment)
        match = match_template_with_mask(
            base_image=image_path,
            template_image=cleaned,
            exclude_regions=[tuple(source_bbox_px)],
        )
        
        target_center_px = match.center
        target_center_pct = self._pixels_to_pct(target_center_px)
        
        return DragAction(
            action="drag",
            source_coordinates=list(source_center_px),
            source_coordinates_pct=list(source_center_pct),
            target_coordinates=[float(target_center_px[0]), float(target_center_px[1])],
            target_coordinates_pct=list(target_center_pct),
            template_match_confidence=match.confidence,
        )
    
    def _solve_text_captcha(self, image_path: str) -> TypeAction:
        """Handle warped-text captchas via LLM text read."""
        text = self.planner.read_text(image_path)
        return TypeAction(action="type", text=text or "")
    
    def _solve_with_legacy_planner(
        self,
        image_path: str,
        context: str,
        include_history: bool,
        elements: Optional[List[dict]],
        prompt_text: Optional[str],
    ) -> CaptchaAction:
        """Preserve the previous two-stage planner flow as a fallback."""
        full_context = context
        if include_history and self._action_history:
            history_str = self._format_history()
            full_context = f"{context}\n\nPrevious actions taken:\n{history_str}"
        
        planned = self.planner.plan(
            image_path,
            full_context,
            elements=elements,
            prompt_text=prompt_text,
        )
        return self._create_action(image_path, planned, elements)
    
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
    
    def _normalize_elements(self, elements: List[dict]) -> List[dict]:
        """Normalize element bboxes to percentage format."""
        if not elements or not self._current_image_size:
            return elements
        
        img_width, img_height = self._current_image_size
        normalized = []
        
        for elem in elements:
            bbox = elem.get('bbox', [])
            if len(bbox) >= 4:
                # Check if already normalized (values <= 1)
                if all(v <= 1 for v in bbox[:4]):
                    normalized.append(elem)
                else:
                    # Convert pixel coords to percentage
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    normalized.append({
                        **elem,
                        'bbox': [
                            x / img_width,
                            y / img_height,
                            (x + w) / img_width,
                            (y + h) / img_height,
                        ]
                    })
            else:
                normalized.append(elem)
        
        return normalized
    
    def _create_action(
        self,
        image_path: str,
        planned: PlannedAction,
        elements: Optional[List[dict]] = None,
    ) -> CaptchaAction:
        """
        Convert a PlannedAction to a CaptchaAction, extracting coordinates if needed.
        """
        if planned.action_type == "click":
            return self._create_click_action(image_path, planned, elements)
        
        elif planned.action_type == "drag":
            return self._create_drag_action(image_path, planned, elements)
        
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
            return self._create_verify_action(image_path, planned, elements)
        
        elif planned.action_type == "done":
            # Return a wait with 0 duration to signal completion
            return WaitAction(action="wait", duration_ms=0)
        
        else:
            raise ValueError(f"Unknown action type: {planned.action_type}")
    
    def _create_click_action(
        self,
        image_path: str,
        planned: PlannedAction,
        elements: Optional[List[dict]] = None,
    ) -> ClickAction:
        """Create a ClickAction with coordinates."""
        
        # If we have element IDs, use their centers
        if planned.target_element_ids and elements:
            all_coords = []
            for elem_id in planned.target_element_ids:
                coords = self._get_element_center(elem_id, elements)
                if coords:
                    all_coords.append(list(coords))
            
            if all_coords:
                print(f"[Stage 2] Using element centers: {all_coords}")
                return ClickAction(
                    action="click",
                    target_ids=planned.target_element_ids,
                    coordinates=all_coords[0] if len(all_coords) == 1 else None,
                    all_coordinates=all_coords if len(all_coords) > 1 else None,
                )
        
        # Fall back to attention-based extraction
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
        return ClickAction(action="click", coordinates=[x, y])
    
    def _create_drag_action(
        self,
        image_path: str,
        planned: PlannedAction,
        elements: Optional[List[dict]] = None,
    ) -> DragAction:
        """Create a DragAction with source and target coordinates."""
        
        source_coords = None
        target_coords = None
        
        # Try element IDs first
        if planned.source_element_id is not None and elements:
            source_coords = self._get_element_center(planned.source_element_id, elements)
        
        if planned.target_element_id is not None and elements:
            target_coords = self._get_element_center(planned.target_element_id, elements)
        
        # Fall back to attention for source
        if source_coords is None:
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
        
        # Fall back to attention for target
        if target_coords is None:
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
        
        return DragAction(
            action="drag",
            source_id=planned.source_element_id,
            source_coordinates=list(source_coords),
            target_id=planned.target_element_id,
            target_coordinates=list(target_coords)
        )
    
    def _create_verify_action(
        self,
        image_path: str,
        planned: PlannedAction,
        elements: Optional[List[dict]] = None,
    ) -> VerifyAction:
        """Create a VerifyAction."""
        target_id = None
        if planned.target_element_ids:
            target_id = planned.target_element_ids[0]
        
        return VerifyAction(
            action="verify",
            target_id=target_id
        )
    
    def _get_element_center(
        self,
        element_id: int,
        elements: List[dict],
    ) -> Optional[tuple]:
        """Get the center coordinates of an element by ID."""
        for elem in elements:
            eid = elem.get('element_id', elem.get('id'))
            if eid == element_id:
                bbox = elem.get('bbox', [])
                if len(bbox) >= 4:
                    # Determine format and calculate center
                    x, y, w_or_x2, h_or_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    
                    # If values are small (< 1), they're percentages in corner format
                    if all(v <= 1.0 for v in bbox[:4]):
                        # Corner format: [x_min, y_min, x_max, y_max]
                        center_x_pct = (x + w_or_x2) / 2
                        center_y_pct = (y + h_or_y2) / 2
                        
                        if self._current_image_size:
                            return (
                                center_x_pct * self._current_image_size[0],
                                center_y_pct * self._current_image_size[1]
                            )
                        return (center_x_pct, center_y_pct)
                    else:
                        # Pixel format: [x, y, width, height]
                        return (x + w_or_x2 / 2, y + h_or_y2 / 2)
        return None
    
    def _format_history(self) -> str:
        """Format action history for context."""
        lines = []
        for i, action in enumerate(self._action_history[-5:], 1):  # Last 5 actions
            if isinstance(action, ClickAction):
                if action.target_ids:
                    lines.append(f"{i}. Clicked elements {action.target_ids}")
                elif getattr(action, "bounding_boxes", None):
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
        elements_getter: Optional[Callable[[], List[dict]]] = None,
        prompt_text: Optional[str] = None,
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
            elements_getter: Optional callable that returns current numbered elements
            prompt_text: The captcha's instruction text
        
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
            
            # Get current elements if provided
            elements = elements_getter() if elements_getter else None
            
            # Get next action
            action = self.solve_step(
                image_path,
                context,
                include_history=True,
                elements=elements,
                prompt_text=prompt_text,
            )
            
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
        elements: Optional[List[dict]] = None,
        prompt_text: Optional[str] = None,
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
            elements: List of numbered elements
            prompt_text: The captcha's instruction text
            max_questions: Maximum number of clarifying questions to ask
        
        Returns:
            Refined CaptchaAction
        """
        image_path = str(Path(image_path).resolve())
        
        # Get initial plan
        planned = self.planner.plan(
            image_path,
            context,
            elements=elements,
            prompt_text=prompt_text,
        )
        
        print(f"[Initial Plan] {planned.action_type}")
        print(f"[Initial Plan] Reasoning: {planned.reasoning}")
        
        # If click action with object detection, refine with questions
        if (planned.action_type == "click" and 
            planned.object_class_to_detect and 
            elements and
            max_questions > 0):
            
            # Ask about each potential element
            questions_asked = 0
            confirmed_ids = []
            
            for elem in elements[:9]:  # Check first 9 elements (typical 3x3 grid)
                if questions_asked >= max_questions:
                    break
                
                elem_id = elem.get('element_id', elem.get('id'))
                question = f"Does element {elem_id} contain a {planned.object_class_to_detect}?"
                
                print(f"[Question] {question}")
                answer = self.planner.question_self(image_path, question)
                print(f"[Answer] {answer}")
                
                # Simple heuristic: if answer contains "yes" or positive indicators
                answer_lower = answer.lower()
                if any(word in answer_lower for word in ['yes', 'correct', 'contains', 'shows', 'has']):
                    confirmed_ids.append(elem_id)
                
                questions_asked += 1
            
            if confirmed_ids:
                print(f"[Refined] Confirmed elements: {confirmed_ids}")
                planned.target_element_ids = confirmed_ids
        
        # Create and return action
        self._current_elements = elements
        if self._current_image_size is None:
            with Image.open(image_path) as img:
                self._current_image_size = img.size
        
        action = self._create_action(image_path, planned, elements)
        self._action_history.append(action)
        
        return action
    
    def reset_history(self):
        """Clear the action history."""
        self._action_history = []
        self._current_elements = None
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
        elements: Optional[List[dict]] = None,
        output_path: Optional[str] = None,
    ):
        """
        Visualize object detections and their mapping to elements.
        """
        detections = self.attention_extractor.detect_objects(image_path, object_class)
        
        if elements:
            # Normalize and map
            with Image.open(image_path) as img:
                self._current_image_size = img.size
            
            normalized = self._normalize_elements(elements)
            detections = self.attention_extractor.map_detections_to_elements(
                detections, normalized, iou_threshold=0.2
            )
        
        return self.attention_extractor.visualize_detections(
            image_path,
            detections,
            output_path or "detections_visualization.png",
            elements=elements
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
