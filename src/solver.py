"""
CaptchaSolver - Simplified captcha solver using LLM planning + vision tools.

Flow:
1. Grid detection - if grid found, use numbered overlay and ask planner to select
2. General solving - let planner use detect/segmentation tools to find targets
3. Drag simulation - iterative refinement with visual feedback

Usage:
    from src import CaptchaSolver

    solver = CaptchaSolver(provider="vllm")
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
        provider: str = "vllm",
        api_key: Optional[str] = None,
    ):
        self.debug = DebugManager(DEBUG)
        
        # Restrict providers to the supported set
        if provider not in {"transformers", "vllm"}:
            raise ValueError(f"Unsupported provider '{provider}'. Supported providers are: 'transformers', 'vllm'.")

        # Set default model based on provider
        if model is None:
            if provider == "vllm":
                model = "Jake-Writer-Jobharvest/qwen3-vl-8b-merged-bf16"
            elif provider == "transformers":
                # ActionPlanner handles the recommended model config for transformers
                pass

        # Validate required parameters
        if provider not in {"transformers", "vllm"} and not api_key:
            # This check is actually redundant now that we only have transformers and vllm, 
            # as neither strictly requires api_key (both are local or tool server).
            # But let's keep it clean.
            pass

        # Setup planner
        planner_kwargs: dict = {
            "backend": provider,
            "model": model,
        }

        planner_kwargs["debug_callback"] = self.debug.log

        self.planner = ActionPlanner(**planner_kwargs)  # type: ignore[arg-type]

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
        media_path: str,
        instruction: str = "",
    ) -> Union[CaptchaAction, Dict[str, Any]]:
        """
        Solve a captcha image or video.
        """
        # Resolve path and get image size
        media_path = str(Path(media_path).resolve())
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media not found: {media_path}")

        # Check if input is video
        is_video = any(media_path.lower().endswith(ext) for ext in [".mp4", ".gif", ".avi"])
        
        # For CV processing, we need a static image. 
        # If it's a video, we extract the first frame.
        cv_image_path = media_path
        if is_video:
            import cv2
            cap = cv2.VideoCapture(media_path)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise ValueError(f"Could not read video from {media_path}")
            
            self._image_size = (frame.shape[1], frame.shape[0])
            cap.release()
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                cv_image_path = tf.name
            self._temp_files.append(cv_image_path)
            cv2.imwrite(cv_image_path, frame)
            self.debug.log(f"Extracted first frame of video: {cv_image_path}")
        else:
            with Image.open(media_path) as img:
                self._image_size = img.size

        # Debug: Save base image/frame
        self.debug.save_image(cv_image_path, "00_base_image.png")

        # 1. Check for grid structure
        grid_boxes = find_grid(cv_image_path)
        if grid_boxes:
            self.debug.log(f"Detected grid with {len(grid_boxes)} cells")
            return self._solve_grid(cv_image_path, instruction, grid_boxes)

        # 2. Check for simple checkbox (lightweight)
        img_w, img_h = self._image_size
        if img_h < 400:
            checkbox_box = find_checkbox(cv_image_path)
            if checkbox_box:
                self.debug.log(f"Detected checkbox at {checkbox_box}")
                x, y, w, h = checkbox_box
                return ClickAction(
                    action="click",
                    target_bounding_boxes=[[x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h]],
                )

        # 3. General solving with tool use
        self.debug.log(f"Using general solving flow (is_video={is_video})")
        return self._solve_general(media_path, instruction, cv_image_path=cv_image_path)

    def solveVideo(self, *args, **kwargs):
        """Alias for solve() to handle videos."""
        return self.solve(*args, **kwargs)

    def _solve_grid(
        self,
        image_path: str,
        instruction: str,
        grid_boxes: List[Tuple[int, int, int, int]],
    ) -> Union[ClickAction, DoneAction, WaitAction]:
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
            cv_selected, cv_loading = detect_selected_cells(image_path, grid_boxes, self.debug)
            if cv_selected:
                self.debug.log(f"CV Detected already selected cells: {cv_selected}")
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
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
            overlay_path = tf.name
        self._temp_files.append(overlay_path)

        try:
            # Prepare overlays - only for selectable cells
            overlays = []
            valid_indices = []
            filtered_out = []
            
            for i, (x1, y1, x2, y2) in enumerate(grid_boxes):
                idx = i + 1
                w = x2 - x1
                h = y2 - y1
                
                # Filter out selected and loading cells from the overlay
                if idx in cv_selected or idx in cv_loading:
                    filtered_out.append(idx)
                    continue
                
                overlays.append(
                    {"bbox": [x1, y1, w, h], "number": idx, "color": "#00FF00", "box_style": "solid"}
                )
                valid_indices.append(idx)
            
            if filtered_out:
                self.debug.log(f"Cells filtered out: {filtered_out}")
            self.debug.log(f"Selectable cells: {valid_indices}")

            # If no cells are valid for selection
            if not overlays:
                if cv_loading:
                    return WaitAction(action="wait", duration_ms=1000)
                else:
                    return DoneAction(action="done")

            add_overlays_to_image(image_path, overlays, output_path=overlay_path, label_position="top-right")
            self.debug.save_image(overlay_path, "01_grid_overlay.png")

            # Ask planner which squares to select
            selected_numbers = self.planner.get_grid_selection(
                overlay_path, 
                rows=rows, 
                cols=cols, 
                instruction=instruction
            )
            
            # 2. Post-processing: Filter out cells that are already selected (using detect_selected_cells results)
            # This ensures we don't click cells that the planner might have hallucinated
            # or that were already marked as selected or loading by CV.
            final_selections = []
            for n in selected_numbers:
                try:
                    val = int(n)
                    # Check against CV detections from the earlier detect_selected_cells call
                    if val in cv_selected:
                        self.debug.log(f"Post-processing: Filtering out cell {val} - already selected (CV detected)")
                        continue
                    if val in cv_loading:
                        self.debug.log(f"Post-processing: Filtering out cell {val} - cell is loading (CV detected)")
                        continue
                    if val not in valid_indices:
                        # This covers cells that are out of range or otherwise invalid
                        self.debug.log(f"Post-processing: Filtering out cell {val} - not in valid overlay indices")
                        continue
                    final_selections.append(val)
                except (ValueError, TypeError):
                    self.debug.log(f"Post-processing: Skipping invalid selection format: {n}")
                    continue
            
            if len(selected_numbers) != len(final_selections):
                filtered = [n for n in selected_numbers if int(n) not in final_selections]
                self.debug.log(f"Post-processing filtered out: {filtered}")
            
            selected_numbers = final_selections
            self.debug.log(f"Final grid selection: {selected_numbers}")

            # Wait Logic
            if not selected_numbers and cv_loading:
                return WaitAction(action="wait", duration_ms=1000)

            if not selected_numbers:
                return DoneAction(action="done")

            # Convert to ClickActions
            target_bounding_boxes = []
            assert self._image_size is not None
            img_w, img_h = self._image_size
            for num in selected_numbers:
                try:
                    idx = int(num) - 1
                except (ValueError, TypeError):
                    self.debug.log(f"Skipping invalid selection: {num}")
                    continue

                if 0 <= idx < len(grid_boxes):
                    x1, y1, x2, y2 = grid_boxes[idx]
                    bbox_pct = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]
                    target_bounding_boxes.append(bbox_pct)

            if not target_bounding_boxes:
                return DoneAction(action="done")

            return ClickAction(
                action="click",
                target_bounding_boxes=target_bounding_boxes,
            )

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
        media_path: str,
        instruction: str,
        cv_image_path: Optional[str] = None,
    ) -> CaptchaAction:
        """
        General solving using planner.
        """
        max_tool_calls = 5
        history: List[str] = []
        cv_path = cv_image_path or media_path
        
        # DO NOT initialize attention here. Let tools do it on-demand to avoid 
        # initializing CUDA before vLLM forks.

        # 1. Pre-detect all objects to provide IDs for the planner
        self.debug.log("Running detect_all to identify objects...")
        attention = self._get_attention()
        # Use more descriptive prompts to catch varied objects
        raw_objects = attention.detect_all(cv_path)
        
        # Assign initial IDs
        self.current_objects = []
        for i, obj in enumerate(raw_objects):
            obj["id"] = i + 1
            # Add a generic label if none exists
            if not obj.get("label"):
                obj["label"] = "item"
            self.current_objects.append(obj)
        self.next_object_id = len(self.current_objects) + 1
        
        self.debug.log(f"Detected {len(self.current_objects)} objects")

        # 2. Create labeled context image for the planner
        ext = os.path.splitext(cv_path)[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
            labeled_path = tf.name
        self._temp_files.append(labeled_path)
        
        shutil.copy2(cv_path, labeled_path)
        if self.current_objects:
            overlays = []
            for obj in self.current_objects:
                overlays.append({
                    "bbox": obj["bbox"],
                    "number": obj["id"],
                    "color": "#FF0000",
                    "box_style": "dashed"
                })
            add_overlays_to_image(labeled_path, overlays)
            self.debug.save_image(labeled_path, "01_detected_objects.png")

        for i in range(max_tool_calls):
            self.debug.log(f"Planning step {i+1}/{max_tool_calls}")
            
            # Use labeled image as context if objects were found
            current_context = labeled_path if self.current_objects else cv_path
            self.debug.save_image(current_context, f"step_{i+1}_context.png")
            
            result = self.planner.plan_with_tools(
                media_path,
                instruction, 
                objects=self.current_objects,
                history=history,
                context_image_path=current_context
            )
            self.debug.log(f"Planner Result: {result}")

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
                new_detections_found = False
                for call in tool_calls:
                    tool = call["name"]
                    args = call.get("args", {})
                    
                    if tool == "detect":
                        object_class = args.get("object_class", "target")
                        max_items = args.get("max_items", 10)
                        self.debug.log(f"Tool call: detect('{object_class}', max_items={max_items})")
                        attention = self._get_attention()
                        detections = detect(attention, media_path, object_class, max_objects=max_items)

                        if detections:
                            new_ids = []
                            for det in detections[:max_items]:
                                det["id"] = self.next_object_id
                                self.next_object_id += 1
                                self.current_objects.append(det)
                                new_ids.append(det["id"])
                            
                            msg = f"detect('{object_class}') found {len(new_ids)} items with IDs: {new_ids}"
                            history.append(f"Tool call: {msg}")
                            self.debug.log(msg)
                            new_detections_found = True
                        else:
                            msg = f"detect('{object_class}') found no items."
                            history.append(f"Tool call: {msg}")
                            self.debug.log(msg)

                    elif tool == "simulate_drag":
                        source_id = args.get("source_id")
                        self.debug.log(f"Tool call: simulate_drag(source_id={source_id})")
                        
                        # Execute drag simulation
                        drag_result = simulate_drag(
                            self,
                            media_path,
                            instruction,
                            primary_goal=result.get("goal") or instruction or "Complete the drag puzzle",
                            source_id=source_id,
                        )
                        return DragAction(**drag_result)

                    elif tool == "find_connected_elems":
                        conn_instruction = args.get("instruction", instruction)
                        self.debug.log(f"Tool call: find_connected_elems('{conn_instruction}')")
                        connected_detections = find_connected_elems(media_path, conn_instruction)
                        
                        if connected_detections:
                            new_ids = []
                            for det in connected_detections:
                                det["id"] = self.next_object_id
                                self.next_object_id += 1
                                self.current_objects.append(det)
                                new_ids.append(det["id"])
                            
                            msg = f"find_connected_elems found {len(new_ids)} items with IDs: {new_ids}"
                            history.append(f"Tool call: {msg}")
                            new_detections_found = True
                        else:
                            history.append("Tool call: find_connected_elems found no items.")

                # If we found new detections, continue the planning loop so the planner can use the new IDs
                if new_detections_found:
                    continue
                
                # If tool calls produced nothing, try next iteration anyway
                continue

            # Handle direct actions
            planner_action = result.get("action")
            if not isinstance(planner_action, dict):
                planner_action = result # Fallback to top-level if "action" is not a dict
                
            action_type = planner_action.get("action") or result.get("action_type")
            if not action_type:
                continue

            history.append(f"Action proposed: {action_type}")

            if action_type == "type":
                text = planner_action.get("text", "")
                target_id = planner_action.get("target_id")
                
                target_bbox = None
                obj = None
                if target_id is not None:
                    try:
                        target_id_int = int(target_id)
                        for o in self.current_objects:
                            if o.get("id") == target_id_int:
                                obj = o
                                break
                    except (ValueError, TypeError):
                        pass
                
                if obj:
                    # Normalized [x1, y1, x2, y2]
                    target_bbox = obj["bbox"] if "bbox" in obj else [obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]]

                return TypeAction(
                    action="type",
                    text=text,
                    target_bounding_box=target_bbox,
                )

            elif action_type == "wait":
                return WaitAction(
                    action="wait",
                    duration_ms=planner_action.get("duration_ms", 500),
                )

            elif action_type == "drag":
                source_id = planner_action.get("source_id")
                target_id = planner_action.get("target_id")
                
                try:
                    source_id = int(source_id) if source_id is not None else None
                    target_id = int(target_id) if target_id is not None else None
                except (ValueError, TypeError):
                    pass

                drag_result = simulate_drag(
                    self,
                    media_path,
                    instruction,
                    primary_goal=result.get("goal") or "Complete the drag puzzle",
                    source_id=source_id,
                    target_id=target_id,
                )
                return DragAction(**drag_result)

            elif action_type == "click":
                target_ids = planner_action.get("target_ids")
                if not target_ids and planner_action.get("target_id"):
                    target_ids = [planner_action["target_id"]]
                
                if not target_ids:
                    return DoneAction(action="done")

                bboxes = []
                for tid in target_ids:
                    try:
                        target_id_int = int(tid)
                    except (ValueError, TypeError):
                        continue

                    obj = None
                    for o in self.current_objects:
                        if o.get("id") == target_id_int:
                            obj = o
                            break
                    
                    if obj:
                        bbox_pct = obj["bbox"] if "bbox" in obj else [obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]]
                        bboxes.append(bbox_pct)
                
                if not bboxes:
                    return DoneAction(action="done")

                return ClickAction(
                    action="click",
                    target_bounding_boxes=bboxes,
                )

            elif action_type == "done":
                return DoneAction(action="done")

        # Fallback after max tool calls
        return DoneAction(action="done")

# Convenience function
def solve_captcha(media_path: str, instruction: str = "", **kwargs) -> Union[CaptchaAction, Dict[str, Any]]:
    """
    Convenience function to solve a captcha image or video.
    """
    solver = CaptchaSolver(**kwargs)
    return solver.solve(media_path, instruction)
