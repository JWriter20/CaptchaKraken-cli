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
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
from PIL import Image

from .action_types import (
    CaptchaAction,
    ClickAction,
    DragAction,
    TypeAction,
    WaitAction,
    DoneAction,
)
from .attention import AttentionExtractor
from .overlay import add_drag_overlay, add_overlays_to_image
from .planner import ActionPlanner
from .grid_planner import GridPlanner
from .image_processor import ImageProcessor
from .timing import timed
from .tool_calls.find_grid import find_grid, detect_selected_cells
from .tool_calls.find_checkbox import find_checkbox
from .tool_calls.segment import segment
from .tool_calls.detect import detect
from .tool_calls.simulate_drag import simulate_drag
from .tool_calls.find_connected_elems import find_connected_elems

# Debug flag - set via CAPTCHA_DEBUG=1 environment variable
DEBUG = os.getenv("CAPTCHA_DEBUG", "0") == "1"


class DebugManager:
    """Manages debug logging and artifacts."""
    def __init__(self, debug_enabled: bool):
        self.enabled = debug_enabled
        # Use absolute path for safety and clarity
        self.base_dir = Path("latestDebugRun").resolve()
        self.log_file = self.base_dir / "log.txt"
        
        if self.enabled:
            self._setup_dir()

    def _setup_dir(self):
        """Clear and recreate the debug directory."""
        if self.base_dir.exists():
            try:
                shutil.rmtree(self.base_dir)
            except Exception as e:
                print(f"[DebugManager] Warning: Could not clear debug dir: {e}", file=sys.stderr)
        
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "w") as f:
                f.write(f"Debug Run Started: {datetime.now()}\n")
        except Exception as e:
            print(f"[DebugManager] Error creating debug dir: {e}", file=sys.stderr)

    def log(self, message: str):
        """Log a message to console and file."""
        # Always print to stderr for immediate feedback during dev
        if self.enabled:
            print(f"[DEBUG] {message}", file=sys.stderr)
            try:
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
            except Exception:
                pass
        elif DEBUG: # Fallback if instance disabled but global env var is set
             print(f"[Solver] {message}", file=sys.stderr)

    def save_image(self, image_path: str, name: str) -> Optional[str]:
        """Save a copy of an image to the debug directory."""
        if not self.enabled:
            return None
            
        target = self.base_dir / name
        try:
            shutil.copy2(image_path, target)
            self.log(f"Saved image: {name}")
            return str(target)
        except Exception as e:
            self.log(f"Failed to save image {name}: {e}")
            return None


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
        self.debug = DebugManager(DEBUG)
        
        # Restrict providers to the supported set
        if provider not in {"ollama", "gemini", "openrouter"}:
            raise ValueError(f"Unsupported provider '{provider}'. Supported providers are: 'ollama', 'gemini', 'openrouter'.")

        # Set default model based on provider
        if model is None:
            if provider == "ollama":
                model = "qwen3-vl:4b"
            elif provider == "gemini":
                model = "gemini-2.5-flash-lite"
            elif provider == "openrouter":
                model = "google/gemini-2.0-flash-lite-preview-02-05:free"

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
        elif provider == "openrouter" and api_key:
            planner_kwargs["openrouter_key"] = api_key

        planner_kwargs["debug_callback"] = self.debug.log

        self.planner = ActionPlanner(**planner_kwargs)  # type: ignore[arg-type]
        self.grid_planner = GridPlanner(**planner_kwargs)

        # IMPORTANT: AttentionExtractor imports/initializes heavyweight deps (torch, HF models).
        # Most reCAPTCHA/hCAPTCHA grid flows do not need it. Lazily initialize on-demand to
        # keep single-shot CLI invocations fast.
        self._attention: Optional[AttentionExtractor] = None

        # Specialized tools (CV utilities). AttentionExtractor is optional and can be injected later.
        self.image_processor = ImageProcessor(None, self.planner, self.debug)

        # Image size cache
        self._image_size: Optional[Tuple[int, int]] = None
        self._temp_files: List[str] = []

    def _get_attention(self) -> AttentionExtractor:
        """
        Lazily construct the AttentionExtractor.

        This avoids importing torch / initializing heavy vision deps for grid-only or checkbox-only flows.
        """
        if self._attention is None:
            self._attention = AttentionExtractor(device=None)  # Auto-detect
            # Keep ImageProcessor in sync for background removal/segmentation code paths
            self.image_processor.attention = self._attention
        return self._attention

    def __del__(self):
        # Cleanup temp files
        for f in self._temp_files:
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception:
                    pass

    def solve(
        self,
        image_path: str,
        instruction: str = "",
    ) -> Union[CaptchaAction, List[ClickAction], Dict[str, Any]]:
        """
        Solve a captcha image or video.

        Args:
            image_path: Path to the captcha image or video
            instruction: Optional instruction text (e.g., "Select all traffic lights")

        Returns:
            CaptchaAction, List[ClickAction], or a dict with actions and metadata
        """
        with timed("solver.solve.total"):
            # Resolve path and get image size
            with timed("solver.resolve_path"):
                image_path = str(Path(image_path).resolve())
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")

            # Check if input is video
            is_video = any(image_path.lower().endswith(ext) for ext in [".mp4", ".webm", ".gif", ".avi"])
            
            # For CV processing, we need a static image. 
            # If it's a video, we extract the first frame.
            cv_image_path = image_path
            if is_video:
                with timed("solver.video.extract_first_frame"):
                    import cv2
                    cap = cv2.VideoCapture(image_path)
                    ret, frame = cap.read()
                    if not ret:
                        cap.release()
                        raise ValueError(f"Could not read video from {image_path}")
                    
                    self._image_size = (frame.shape[1], frame.shape[0])
                    cap.release()
                    
                    # Save first frame to temp file for CV tools
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                        cv_image_path = tf.name
                    self._temp_files.append(cv_image_path)
                    cv2.imwrite(cv_image_path, frame)
                    self.debug.log(f"Extracted first frame of video for CV processing: {cv_image_path}")
            else:
                with timed("solver.open_image"):
                    with Image.open(image_path) as img:
                        self._image_size = img.size

            # Debug: Save base image/frame
            with timed("solver.debug.save_base_image"):
                self.debug.save_image(cv_image_path, "00_base_image.png")

            # 1. Check for grid structure
            with timed("solver.grid.detect_grid_boxes"):
                grid_boxes = find_grid(cv_image_path)
            if grid_boxes:
                self.debug.log(f"Detected grid with {len(grid_boxes)} cells")
                # For grid, we usually just need the static image, but we can pass the video path
                # to the grid planner if needed. For now, use the static image for grid.
                return self._solve_grid(cv_image_path, instruction, grid_boxes)

            # 2. Check for simple checkbox (lightweight)
            img_w, img_h = self._image_size
            if img_h < 400:
                with timed("solver.checkbox.detect_checkbox"):
                    checkbox_box = find_checkbox(cv_image_path)
                if checkbox_box:
                    self.debug.log(f"Detected checkbox at {checkbox_box}")
                    x, y, w, h = checkbox_box
                    # Convert to normalized click action
                    return ClickAction(
                        action="click",
                        target_bounding_box=[x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h],
                    )

            # 3. General solving with tool use
            # Pass the ORIGINAL image_path (could be video) to _solve_general
            self.debug.log(f"Using general solving flow (is_video={is_video})")
            return self._solve_general(image_path, instruction, cv_image_path=cv_image_path)

    def _solve_grid(
        self,
        image_path: str,
        instruction: str,
        grid_boxes: List[Tuple[int, int, int, int]],
    ) -> Union[List[ClickAction], DoneAction, WaitAction]:
        """
        Solve grid selection captcha.
        """
        n = len(grid_boxes)
        if n == 9:
            rows, cols = 3, 3
        elif n == 16:
            rows, cols = 4, 4
        else:
            cols = int(math.sqrt(n))
            rows = math.ceil(n / cols)

        self.debug.log(f"_solve_grid called with {n} cells ({rows}x{cols})")
        self.debug.log(f"Instruction passed to planner: '{instruction}'")

        # 1. Detect state (Selected / Loading) via CV
        cv_selected = []
        cv_loading = []
        try:
            with timed("solver.grid.cv_detect_selected_cells", extra=f"cells={len(grid_boxes)}"):
                cv_selected, cv_loading = detect_selected_cells(image_path, grid_boxes, self.debug)
            if cv_selected:
                self.debug.log(f"CV Detected already selected cells: {cv_selected}")
                # We don't return DoneAction immediately anymore, to allow for multi-round selection
                # if the puzzle requires it. The selected cells will be filtered out of the overlay.
            else:
                self.debug.log("CV Detection: No selected cells detected")
            if cv_loading:
                self.debug.log(f"CV Detected loading cells: {cv_loading}")
            else:
                self.debug.log("CV Detection: No loading cells detected")
        except Exception as e:
            self.debug.log(f"Error in CV check for selected cells: {e}")

        # Create temp image with numbered overlay
        ext = os.path.splitext(image_path)[1] or ".png"
        with timed("solver.grid.create_temp_overlay_path"):
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                overlay_path = tf.name
        self._temp_files.append(overlay_path)

        try:
            # Prepare overlays - only for selectable cells
            overlays = []
            valid_indices = []
            filtered_out = []
            
            with timed("solver.grid.build_overlay_boxes", extra=f"cells={len(grid_boxes)}"):
                for i, (x1, y1, x2, y2) in enumerate(grid_boxes):
                    idx = i + 1
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Filter out selected and loading cells from the overlay
                    if idx in cv_selected or idx in cv_loading:
                        filtered_out.append(idx)
                        self.debug.log(f"Filtering out cell {idx} from overlay (selected={idx in cv_selected}, loading={idx in cv_loading})")
                        continue
                    
                    # Tile-grid overlay: match the coordinate grid overlay aesthetic (solid green + black underlay).
                    # This is less visually confusing than multi-color dashed boxes on busy captcha images.
                    overlays.append(
                        {"bbox": [x1, y1, w, h], "number": idx, "color": "#00FF00", "box_style": "solid"}
                    )
                    valid_indices.append(idx)
            
            if filtered_out:
                self.debug.log(f"Total cells filtered out: {filtered_out}")
            self.debug.log(f"Cells available for selection (numbered in overlay): {valid_indices}")

            # If no cells are valid for selection
            if not overlays:
                if cv_loading:
                    self.debug.log("No selectable cells, but loading cells present. Waiting.")
                    return WaitAction(action="wait", duration_ms=1000)
                else:
                    self.debug.log("No selectable cells found and no loading cells. Assuming Done.")
                    return DoneAction(action="done")

            with timed("solver.grid.render_overlay_image"):
                add_overlays_to_image(image_path, overlays, output_path=overlay_path, label_position="top-right")
            with timed("solver.debug.save_grid_overlay"):
                self.debug.save_image(overlay_path, "01_grid_overlay.png")

            # Ask planner which squares to select
            with timed("solver.grid.llm_get_grid_selection"):
                selected_numbers = self.grid_planner.get_grid_selection(
                    overlay_path, 
                    rows=rows, 
                    cols=cols, 
                    instruction=instruction
                )
            
            # Filter just in case the model hallucinates numbers not in overlay
            # (The model shouldn't see numbers that aren't there, but good for safety)
            before_filter = selected_numbers.copy()
            with timed("solver.grid.filter_llm_numbers"):
                # Ensure they are ints and in valid_indices
                converted_numbers = []
                for n in selected_numbers:
                    try:
                        val = int(n)
                        if val in valid_indices:
                            converted_numbers.append(val)
                    except (ValueError, TypeError):
                        continue
                selected_numbers = converted_numbers
            
            if len(before_filter) != len(selected_numbers):
                filtered = set(before_filter) - set(selected_numbers)
                self.debug.log(f"WARNING: LLM selected cells that were filtered out: {filtered}")
                self.debug.log(f"These cells should not have been in the overlay (CV detected them as selected/loading)")

            self.debug.log(f"Grid selection (after filtering): {selected_numbers}")

            # Wait Logic
            if not selected_numbers and cv_loading:
                self.debug.log("No new selections and loading cells detected. Returning WaitAction.")
                return WaitAction(action="wait", duration_ms=1000)

            if not selected_numbers:
                return DoneAction(action="done")

            # Convert to ClickActions
            actions = []
            assert self._image_size is not None
            img_w, img_h = self._image_size
            with timed("solver.grid.build_click_actions", extra=f"n={len(selected_numbers)}"):
                for num in selected_numbers:
                    try:
                        idx = int(num) - 1
                    except (ValueError, TypeError):
                        self.debug.log(f"Skipping invalid selection: {num}")
                        continue

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

            if not actions:
                return DoneAction(action="done")

            return actions

        finally:
            if os.path.exists(overlay_path):
                os.unlink(overlay_path)

    def _get_object_by_id(self, target: str, objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Try to find an object by ID in the target string (e.g. "item 5")."""
        import re
        # Look for "item N", "number N", "#N", or just "N" if it's the whole string
        match = re.search(r"(?:item|number|#)?\s*(\d+)", target, re.IGNORECASE)
        if match:
            try:
                idx = int(match.group(1))
                for obj in objects:
                    if obj.get("id") == idx:
                        return obj
            except ValueError:
                pass
        return None

    def _solve_general(
        self,
        image_path: str,
        instruction: str,
        cv_image_path: Optional[str] = None,
    ) -> Union[CaptchaAction, List[ClickAction]]:
        """
        General solving using planner with detect/point tools.
        """
        max_tool_calls = 5
        history: List[str] = []
        cv_path = cv_image_path or image_path

        # General solving relies on detect/point/segmentation -> needs heavyweight vision stack.
        attention = self._get_attention()
        
        # Run initial segmentation immediately (use static image)
        current_cv_image_path, current_objects = segment(self.image_processor, attention, cv_path, self.debug)
        if current_objects:
            history.append(f"Initial segmentation found {len(current_objects)} objects.")

        for i in range(max_tool_calls):
            self.debug.log(f"Planning step {i+1}/{max_tool_calls}")
            # IMPORTANT: Pass the ORIGINAL image_path (could be video) to the planner
            # but use the segmented/labeled static image (current_cv_image_path) for context
            result = self.planner.plan_with_tools(
                image_path, # Could be video
                instruction, 
                objects=current_objects,
                history=history,
                context_image_path=current_cv_image_path # Static image with labels
            )

            # Log analysis
            if "analysis" in result:
                self.debug.log(f"Analysis: {result['analysis']}")
            if "goal" in result:
                self.debug.log(f"Goal: {result['goal']}")

            # Collect actions from all tool calls
            collected_actions: List[ClickAction] = []
            
            # Normalize single tool_call to list format for processing
            tool_calls = result.get("tool_calls", [])
            if not tool_calls and result.get("tool_call"):
                tool_calls = [result["tool_call"]]
            
            # Process all requested tools
            if tool_calls:
                for call in tool_calls:
                    tool = call["name"]
                    args = call.get("args", {})
                    history.append(f"Tool call: {tool}({args})")
                    
                    if tool == "detect":
                        object_class = args.get("object_class", "target")
                        self.debug.log(f"Tool call: detect('{object_class}')")
                        detections = detect(attention, current_cv_image_path, object_class)

                        if detections:
                            for det in detections:
                                bbox = det["bbox"]
                                collected_actions.append(
                                    ClickAction(
                                        action="click",
                                        target_bounding_box=bbox,
                                    )
                                )
                        else:
                            # Fallback to detect with max_objects=1
                            self.debug.log("No detections, trying fallback...")
                            fallback_detections = attention.detect(current_cv_image_path, object_class, max_objects=1)
                            if fallback_detections:
                                obj = fallback_detections[0]
                                x = (obj["x_min"] + obj["x_max"]) / 2
                                y = (obj["y_min"] + obj["y_max"]) / 2
                                collected_actions.append(
                                    ClickAction(
                                        action="click",
                                        target_coordinates=[x, y],
                                    )
                                )
                            else:
                                collected_actions.append(
                                    ClickAction(
                                        action="click",
                                        target_coordinates=[0.5, 0.5],
                                    )
                                )

                    elif tool == "simulate_drag":
                        source = args.get("source", "movable item")
                        target_hint = args.get("target_hint", "matching slot")
                        target_object_id = args.get("target_object_id")
                        source_quality = args.get("source_quality")
                        refined_source = args.get("refined_source")
                        location_hint = args.get("location_hint")

                        self.debug.log(f"Tool call: simulate_drag('{source}', '{target_hint}')")
                        
                        # 0. Adjust location_hint if target_object_id is provided
                        if target_object_id:
                            # Try to treat as int, or string "Object N"
                            tgt_obj = self._get_object_by_id(str(target_object_id), current_objects)
                            if tgt_obj:
                                t_bbox = tgt_obj["bbox"]
                                img_w, img_h = self._image_size
                                t_cx = (t_bbox[0] + t_bbox[2] / 2) / img_w
                                t_cy = (t_bbox[1] + t_bbox[3] / 2) / img_h
                                location_hint = [t_cx, t_cy]
                        
                        source_bbox = None
                        obj = self._get_object_by_id(source, current_objects)
                        if obj:
                            b = obj["bbox"]
                            source_bbox = [b[0], b[1], b[0] + b[2], b[1] + b[3]]

                        # Execute drag simulation
                        description_to_use = refined_source if refined_source else source
                        drag_result = simulate_drag(
                            self,
                            cv_path,
                            instruction,
                            description_to_use,
                            target_hint,
                            source_bbox_override=source_bbox,
                            initial_location_hint=location_hint,
                            primary_goal=result.get("goal"),
                        )
                        return DragAction(**drag_result)

                    elif tool == "find_connected_elems":
                        conn_instruction = args.get("instruction", instruction)
                        self.debug.log(f"Tool call: find_connected_elems('{conn_instruction}')")
                        connected_detections = find_connected_elems(cv_path, conn_instruction)
                        
                        if connected_detections:
                            for det in connected_detections:
                                collected_actions.append(
                                    ClickAction(
                                        action="click",
                                        target_bounding_box=det.get("bbox"),
                                    )
                                )
                        else:
                            # If not implemented or no results, fallback to center click
                            collected_actions.append(
                                ClickAction(
                                    action="click",
                                    target_coordinates=[0.5, 0.5],
                                )
                            )

                # Return collected actions if any
                if collected_actions:
                    if len(collected_actions) == 1:
                        return collected_actions[0]
                    return collected_actions
                
                # If tool calls produced nothing (and no segmentation), try next iteration
                continue

            # Handle direct actions
            action_type = result.get("action_type", "click")
            history.append(f"Action proposed: {action_type}")

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
                source_desc = result.get("source_description")
                target_desc = result.get("drag_target_description") or result.get("target_description")
                
                # Optimization: If both source and target are identified objects, return direct action
                source_obj = self._get_object_by_id(source_desc, current_objects)
                target_obj = self._get_object_by_id(target_desc, current_objects)
                
                if source_obj and target_obj:
                    self.debug.log(f"Direct drag from Object {source_obj['id']} to Object {target_obj['id']}")
                    img_w, img_h = self._image_size
                    s_bbox = source_obj["bbox"]
                    t_bbox = target_obj["bbox"]
                    s_center = [(s_bbox[0] + s_bbox[2] / 2) / img_w, (s_bbox[1] + s_bbox[3] / 2) / img_h]
                    t_center = [(t_bbox[0] + t_bbox[2] / 2) / img_w, (t_bbox[1] + t_bbox[3] / 2) / img_h]
                    
                    return DragAction(action="drag", source_coordinates=s_center, target_coordinates=t_center)

                drag_result = simulate_drag(
                    self,
                    current_cv_image_path,
                    instruction,
                    source_desc,
                    target_desc,
                    primary_goal=result.get("goal"),
                )
                return DragAction(**drag_result)

            elif action_type == "click":
                target = result.get("target_description", "target")
                
                # Check if target is an object ID
                obj = self._get_object_by_id(target, current_objects)
                if obj:
                    img_w, img_h = self._image_size if self._image_size else Image.open(image_path).size
                    bbox = obj["bbox"] # [x, y, w, h]
                    # Normalize to [x1, y1, x2, y2]
                    bbox_pct = [
                        bbox[0] / img_w,
                        bbox[1] / img_h,
                        (bbox[0] + bbox[2]) / img_w,
                        (bbox[1] + bbox[3]) / img_h
                    ]
                    return ClickAction(
                        action="click",
                        target_bounding_box=bbox_pct,
                    )
                else:
                    detections = attention.detect(current_image_path, target, max_objects=1)
                    if detections:
                        obj = detections[0]
                        x = (obj["x_min"] + obj["x_max"]) / 2
                        y = (obj["y_min"] + obj["y_max"]) / 2
                    else:
                        x, y = 0.5, 0.5
                    
                    return ClickAction(
                        action="click",
                        target_coordinates=[x, y],
                    )

            elif action_type == "done":
                return DoneAction(action="done")

        # Fallback after max tool calls
        return DoneAction(action="done")

# Convenience function
def solve_captcha(image_path: str, instruction: str = "", **kwargs) -> Union[CaptchaAction, List[ClickAction]]:
    """
    Convenience function to solve a captcha.
    """
    solver = CaptchaSolver(**kwargs)
    return solver.solve(image_path, instruction)
