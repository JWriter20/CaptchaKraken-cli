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
        if provider not in {"ollama", "gemini"}:
            raise ValueError(f"Unsupported provider '{provider}'. Supported providers are: 'ollama', 'gemini'.")

        # Set default model based on provider
        if model is None:
            if provider == "ollama":
                model = "qwen3-vl:4b"
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

        planner_kwargs["debug_callback"] = self.debug.log

        self.planner = ActionPlanner(**planner_kwargs)  # type: ignore[arg-type]
        self.grid_planner = GridPlanner(**planner_kwargs)

        # Setup attention extractor (GroundingDINO + CLIP + FastSAM for detect/point)
        self.attention = AttentionExtractor(
            device=None,  # Auto-detect
        )

        # specialized tools
        self.image_processor = ImageProcessor(self.attention, self.planner, self.debug)

        # Image size cache
        self._image_size: Optional[Tuple[int, int]] = None
        self._temp_files: List[str] = []

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
            
        # Debug: Save base image
        self.debug.save_image(image_path, "00_base_image.png")

        # 1. Check for grid structure
        grid_boxes = self.image_processor.get_grid_bounding_boxes(image_path)
        if grid_boxes:
            self.debug.log(f"Detected grid with {len(grid_boxes)} cells")
            return self._solve_grid(image_path, instruction, grid_boxes)

        # 2. Check for simple checkbox (lightweight)
        # Optimization: Only run fast checkbox detection on small/wide images (likely the checkbox widget).
        # Large/tall images are likely challenge windows where we should use the planner to avoid false positives (like footer icons).
        img_w, img_h = self._image_size
        if img_h < 400:
            checkbox_box = self.image_processor.detect_checkbox(image_path)
            if checkbox_box:
                self.debug.log(f"Detected checkbox at {checkbox_box}")
                x, y, w, h = checkbox_box
                # Convert to normalized click action
                return ClickAction(
                    action="click",
                    target_bounding_box=[x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h],
                )

        # 3. General solving with tool use
        self.debug.log("Using general solving flow")
        return self._solve_general(image_path, instruction)

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
            cv_selected, cv_loading = self.image_processor.detect_selected_cells(image_path, grid_boxes)
            if cv_selected:
                self.debug.log(f"CV Detected already selected cells: {cv_selected}")
                # For grids most of the time we can select all of the tiles in one go, doing more often 
                # confuses the model and leads it to select extraneous tiles. Since there are selected cells detected
                # we know this is not one of the fading-in puzzles, so we can assume the first run through solved the puzzle.
                return DoneAction(action="done")
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
                    self.debug.log(f"Filtering out cell {idx} from overlay (selected={idx in cv_selected}, loading={idx in cv_loading})")
                    continue
                
                # Use a high-contrast color that doesn't resemble traffic lights (Red/Yellow/Green)
                # or Checkmarks (Blue). Magenta/Purple is a good choice.
                overlays.append({"bbox": [x1, y1, w, h], "number": idx, "color": "#9B59B6"})
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

            add_overlays_to_image(image_path, overlays, output_path=overlay_path, label_position="top-right")
            self.debug.save_image(overlay_path, "01_grid_overlay.png")

            # Ask planner which squares to select
            selected_numbers = self.grid_planner.get_grid_selection(
                overlay_path, 
                rows=rows, 
                cols=cols, 
                instruction=instruction
            )
            
            # Filter just in case the model hallucinates numbers not in overlay
            # (The model shouldn't see numbers that aren't there, but good for safety)
            before_filter = selected_numbers.copy()
            selected_numbers = [n for n in selected_numbers if n in valid_indices]
            
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

    def _handle_segmentation(self, image_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Sharpen image, segment with SAM2 via detectInteractable, and create numbered overlay.
        Returns (path to new image, list of objects found).
        """
        self.debug.log("Running segmentation and labeling...")
        
        ext = os.path.splitext(image_path)[1] or ".png"
        
        # 1. Pre-process: Merge similar colors
        with tempfile.NamedTemporaryFile(suffix=f"_merged{ext}", delete=False) as tf:
            merged_path = tf.name
        self._temp_files.append(merged_path)
        
        sharpen_input = image_path
        try:
            self.debug.log("Merging similar colors...")
            self.image_processor.merge_similar_colors(image_path, merged_path, k=12)
        except Exception as e:
            self.debug.log(f"Color merging failed: {e}")

        # 2. Greyscale image
        with tempfile.NamedTemporaryFile(suffix=f"_greyscale{ext}", delete=False) as tf:
            greyscale_path = tf.name
        self._temp_files.append(greyscale_path)
        try:
            self.image_processor.to_greyscale(merged_path, greyscale_path)
        except Exception as e:
            self.debug.log(f"Greyscale conversion failed: {e}")
            shutil.copy2(merged_path, greyscale_path)
            sharpen_input = greyscale_path
        else:
            sharpen_input = merged_path

        # 3. Sharpen image
        with tempfile.NamedTemporaryFile(suffix=f"_sharp{ext}", delete=False) as tf:
            sharp_path = tf.name
        self._temp_files.append(sharp_path)
        
        try:
            self.image_processor.sharpen_image(sharpen_input, sharp_path)
        except Exception as e:
            self.debug.log(f"Sharpening failed: {e}")
            shutil.copy2(sharpen_input, sharp_path)

        # 2. Segment with SAM2 using detectInteractable (better sensitivity/filtering)
        # detectInteractable returns objects with pixel bboxes [x, y, w, h]
        result = self.attention.detectInteractable(sharp_path)
        objects = result.get("objects", [])
        
        if not objects:
            self.debug.log("No objects found during segmentation.")
            return image_path, []

        # 3. Create overlays
        overlays = []
        # Use high-contrast colors
        colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F1C40F", "#9B59B6", "#E67E22"]
        
        # Keep track of objects with ID for later lookup
        labeled_objects = []
        
        for i, obj in enumerate(objects):
            bbox = obj["bbox"] # [x, y, w, h]
            
            # Create overlay item
            overlays.append({
                "bbox": bbox,
                "number": i + 1,
                "color": colors[i % len(colors)]
            })
            
            # Store with ID
            obj_with_id = obj.copy()
            obj_with_id["id"] = i + 1
            labeled_objects.append(obj_with_id)

        # 4. Save overlay image
        with tempfile.NamedTemporaryFile(suffix=f"_labeled{ext}", delete=False) as tf:
            labeled_path = tf.name
        self._temp_files.append(labeled_path)
        
        add_overlays_to_image(image_path, overlays, output_path=labeled_path)
        self.debug.log(f"Segmented {len(overlays)} objects.")
        self.debug.save_image(labeled_path, "02_segmented_overlay.png")
        
        return labeled_path, labeled_objects

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
    ) -> Union[CaptchaAction, List[ClickAction]]:
        """
        General solving using planner with detect/point tools.
        """
        max_tool_calls = 5
        history: List[str] = []
        
        # Run initial segmentation immediately
        current_image_path, current_objects = self._handle_segmentation(image_path)
        if current_objects:
            history.append(f"Initial segmentation found {len(current_objects)} objects.")

        for i in range(max_tool_calls):
            self.debug.log(f"Planning step {i+1}/{max_tool_calls}")
            result = self.planner.plan_with_tools(
                current_image_path, 
                instruction, 
                objects=current_objects,
                history=history
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
                        detections = self.attention.detect_objects(current_image_path, object_class)

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
                            # Fallback to point
                            self.debug.log("No detections, falling back to point()")
                            x, y = self.attention.extract_coordinates(current_image_path, object_class)
                            collected_actions.append(
                                ClickAction(
                                    action="click",
                                    target_coordinates=[x, y],
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
                        if target_object_id:
                            self.debug.log(f"  Target Object ID: {target_object_id}")
                        if source_quality:
                            self.debug.log(f"  Quality: {source_quality}, Refined: {refined_source}")
                        if location_hint:
                            self.debug.log(f"  Location Hint: {location_hint}")
                        
                        source_bbox = None
                        
                        # 0. Adjust location_hint if target_object_id is provided
                        if target_object_id:
                            # Try to treat as int, or string "Object N"
                            tgt_obj = self._get_object_by_id(str(target_object_id), current_objects)
                            if tgt_obj:
                                # Use center of target object as the starting location hint
                                t_bbox = tgt_obj["bbox"] # [x, y, w, h]
                                img_w, img_h = self._image_size
                                
                                # Calculate center in normalized coords
                                t_cx = (t_bbox[0] + t_bbox[2] / 2) / img_w
                                t_cy = (t_bbox[1] + t_bbox[3] / 2) / img_h
                                
                                location_hint = [t_cx, t_cy]
                                self.debug.log(f"Using center of Target Object {tgt_obj['id']} as start: {location_hint}")
                            else:
                                self.debug.log(f"Target object ID '{target_object_id}' not found.")

                        # 1. Check if source is an object ID
                        obj = self._get_object_by_id(source, current_objects)
                        
                        if obj and source_quality == "includes-source":
                            self.debug.log(f"Object {obj.get('id')} contains source. Cropping and refining...")
                            try:
                                with Image.open(image_path) as img:
                                    b = obj["bbox"] # [x, y, w, h]
                                    # Ensure crop box is valid
                                    crop_box = (
                                        int(b[0]), 
                                        int(b[1]), 
                                        int(b[0] + b[2]), 
                                        int(b[1] + b[3])
                                    )
                                    crop_img = img.crop(crop_box)
                                    
                                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                                        crop_path = tf.name
                                    self._temp_files.append(crop_path)
                                    crop_img.save(crop_path)
                                    
                                    search_term = refined_source or source
                                    # Run detect on the crop
                                    detect_result = self.attention.detect(crop_path, search_term, max_objects=1)
                                    detections = detect_result.get("objects", [])
                                    
                                    if detections:
                                        # Convert from {x_min, y_min, x_max, y_max} to [x1, y1, x2, y2]
                                        obj = detections[0]
                                        d = [obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]] # [x1, y1, x2, y2] normalized to CROP
                                        cw, ch = crop_img.size
                                        
                                        # Convert to pixel coords in CROP
                                        cx1 = d[0] * cw
                                        cy1 = d[1] * ch
                                        cx2 = d[2] * cw
                                        cy2 = d[3] * ch
                                        
                                        # Translate to full image
                                        fx1 = cx1 + b[0]
                                        fy1 = cy1 + b[1]
                                        fx2 = cx2 + b[0]
                                        fy2 = cy2 + b[1]
                                        
                                        source_bbox = [fx1, fy1, fx2, fy2]
                                        self.debug.log(f"Refined source bbox: {source_bbox}")
                                    else:
                                        self.debug.log("Refinement failed in crop. Falling back to object bbox.")
                                        source_bbox = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
                                        
                            except Exception as e:
                                self.debug.log(f"Error during crop refinement: {e}")
                                # Fallback
                                b = obj["bbox"]
                                source_bbox = [b[0], b[1], b[0] + b[2], b[1] + b[3]]

                        elif obj:
                            # Default case: "properly-cropped" or unspecified
                            self.debug.log(f"Source identified as existing Object {obj.get('id')} (Assuming Tight)")
                            b = obj["bbox"]
                            source_bbox = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
                        
                        if not source_bbox:
                            # 2. Run detect (either not found, or explicit "not-bounded")
                            search_term = refined_source or source
                            self.debug.log(f"Running detect for '{search_term}'...")
                            
                            # Run detect on CLEAN image
                            detect_result = self.attention.detect(image_path, search_term, max_objects=1)
                            detections = detect_result.get("objects", [])
                            
                            if detections:
                                obj = detections[0]
                                self.debug.log(f"detect found: {obj}")
                                
                                # Convert to visualization format (bbox as [x_min, y_min, x_max, y_max])
                                vis_detections = [{
                                    "bbox": [obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]],
                                    "label": obj.get("label", search_term),
                                    "score": obj.get("score", 0.0)
                                }]
                                
                                # Visualize this detection
                                debug_det_path = str(self.debug.base_dir / "03_detected_source.png")
                                self.attention.visualize_detections(
                                    image_path, 
                                    vis_detections, 
                                    output_path=debug_det_path
                                )
                                self.debug.log(f"Saved detection visualization to {debug_det_path}")
                                
                                # Convert normalized bbox to pixels [x1, y1, x2, y2]
                                img_w, img_h = self._image_size
                                source_bbox = [
                                    obj["x_min"] * img_w,
                                    obj["y_min"] * img_h,
                                    obj["x_max"] * img_w,
                                    obj["y_max"] * img_h
                                ]
                            else:
                                self.debug.log("detect failed to find source.")
                        
                        # Execute drag simulation immediately and return result
                        # NOTE: Use image_path (clean) for drag simulation to avoid artifacts from labels
                        # Use refined_source for the description if available, as it's more descriptive than "Object N"
                        description_to_use = refined_source if refined_source else source
                        
                        return self._solve_drag(
                            image_path,
                            instruction,
                            description_to_use,
                            target_hint,
                            source_bbox_override=source_bbox,
                            initial_location_hint=location_hint,
                            primary_goal=result.get("goal"),
                        )
                            
                    elif tool == "point":
                        target = args.get("target", "target")
                        self.debug.log(f"Tool call: point('{target}')")
                        
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
                            collected_actions.append(
                                ClickAction(
                                    action="click",
                                    target_bounding_box=bbox_pct,
                                )
                            )
                        else:
                            x, y = self.attention.extract_coordinates(current_image_path, target)
                            collected_actions.append(
                                ClickAction(
                                    action="click",
                                    target_coordinates=[x, y],
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
                    
                    # Calculate centers (normalized)
                    # bboxes are [x, y, w, h] in pixels
                    s_center = [
                        (s_bbox[0] + s_bbox[2] / 2) / img_w,
                        (s_bbox[1] + s_bbox[3] / 2) / img_h
                    ]
                    t_center = [
                        (t_bbox[0] + t_bbox[2] / 2) / img_w,
                        (t_bbox[1] + t_bbox[3] / 2) / img_h
                    ]
                    
                    return DragAction(
                        action="drag",
                        source_coordinates=s_center,
                        target_coordinates=t_center
                    )

                return self._solve_drag(
                    current_image_path, # Use current image path (might be segmented)
                    instruction,
                    source_desc,
                    target_desc,
                    primary_goal=result.get("goal"),
                )

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
                    x, y = self.attention.extract_coordinates(current_image_path, target)
                    return ClickAction(
                        action="click",
                        target_coordinates=[x, y],
                    )

            elif action_type == "done":
                return DoneAction(action="done")

        # Fallback after max tool calls
        return DoneAction(action="done")

    def _remove_background(self, image_path: str, prompt: str) -> Optional[Image.Image]:
        """
        Remove background from an image (crop) by selecting the segment matching the prompt.
        Returns RGBA image with transparent background.
        """
        return self.image_processor.remove_background(
            image_path,
            prompt,
            k=5,
            merge_components=False,
            min_area_ratio=0.005,
            pre_merge_colors=True
        )

    def _solve_drag(
        self,
        image_path: str,
        instruction: str,
        source_description: Optional[str],
        target_description: Optional[str],
        max_iterations: int = 5,
        source_bbox_override: Optional[List[float]] = None,
        initial_location_hint: Optional[List[float]] = None,
        primary_goal: Optional[str] = None,
    ) -> DragAction:
        """
        Solve drag puzzle with iterative refinement.
        """
        source_desc = source_description or "movable item"
        assert self._image_size is not None
        img_w, img_h = self._image_size

        # 1. Find source
        if source_bbox_override:
            self.debug.log(f"Using provided source bbox: {source_bbox_override}")
            source_bbox_px = source_bbox_override
            source_x = (source_bbox_px[0] + source_bbox_px[2]) / 2 / img_w
            source_y = (source_bbox_px[1] + source_bbox_px[3]) / 2 / img_h
        else:
            self.debug.log(f"Finding drag source: '{source_desc}'")
            # Try to find the specific item
            detections = self.attention.detect_objects(image_path, source_desc, max_objects=1)
            if detections:
                bbox = detections[0]["bbox"] # [x_min, y_min, x_max, y_max]
                source_x = (bbox[0] + bbox[2]) / 2
                source_y = (bbox[1] + bbox[3]) / 2
                source_bbox_px = [
                    bbox[0] * img_w,
                    bbox[1] * img_h,
                    bbox[2] * img_w,
                    bbox[3] * img_h
                ]
            else:
                # Fallback to point() if detect fails (returns center only)
                source_x, source_y = self.attention.extract_coordinates(image_path, source_desc)
                # Make a best-guess box around the point (e.g. 10% of image)
                box_size = 0.1
                source_bbox_px = [
                    (source_x - box_size/2) * img_w,
                    (source_y - box_size/2) * img_h,
                    (source_x + box_size/2) * img_w,
                    (source_y + box_size/2) * img_h
                ]

        source_coords = [source_x, source_y]
        self.debug.log(f"Drag source at: ({source_x:.2%}, {source_y:.2%})")

        # --- NEW LOGIC START ---
        # Prepare foreground image (remove background from source) using simplified color segmentation
        foreground_image = None
        try:
            # Crop source
            x1, y1, x2, y2 = map(int, source_bbox_px)
            with Image.open(image_path) as img:
                w, h = img.size
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                if x2 > x1 and y2 > y1:
                    source_crop = img.crop((x1, y1, x2, y2))
                    
                    # Save temp crop
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                        crop_path = tf.name
                    self._temp_files.append(crop_path)
                    source_crop.save(crop_path)
                    
                    # Use ImageProcessor
                    # Settings from original code: k=3, merge=True, min_area=0.01
                    foreground_image = self.image_processor.remove_background(
                        crop_path, 
                        prompt=source_desc,
                        k=3,
                        merge_components=True,
                        min_area_ratio=0.01
                    )
                    
                    if foreground_image:
                         self.debug.log("Successfully created foreground image for drag source.")
                    else:
                         self.debug.log("Background removal failed or returned no image.")
        except Exception as e:
            self.debug.log(f"Failed to prepare foreground image: {e}")
        # --- NEW LOGIC END ---

        # 2. Initial target estimate
        # Force start at center (0.5, 0.5) to ensure the model actively looks for the target
        # Update: If initial_location_hint is provided (e.g. from target object ID), use it
        target_x, target_y = 0.5, 0.5
        
        target_found_via_hint = False
        if initial_location_hint:
            target_x, target_y = initial_location_hint
            target_found_via_hint = True
            self.debug.log(f"Using provided initial location hint: ({target_x:.2%}, {target_y:.2%})")
        
        # We only override if we have a robust detection of the target already
        
        if not target_found_via_hint and target_description and target_description != "matching slot":
             self.debug.log(f"Searching for drag target based on description: '{target_description}'")
             # Try detect first
             detections = self.attention.detect_objects(image_path, target_description, max_objects=1)
             if detections:
                 b = detections[0]["bbox"] # [x_min, y_min, x_max, y_max]
                 
                 # Calculate center from [x1, y1, x2, y2]
                 cx = (b[0] + b[2]) / 2
                 cy = (b[1] + b[3]) / 2
                 target_x, target_y = cx, cy
                 self.debug.log(f"Found target candidate at ({target_x:.2f}, {target_y:.2f})")
             else:
                 # Fallback to point (CLIP/GradCAM)
                 try:
                    tx, ty = self.attention.extract_coordinates(image_path, target_description)
                    target_x, target_y = tx, ty
                    self.debug.log(f"CLIP extracted coordinates for target: ({target_x:.2f}, {target_y:.2f})")
                 except Exception as e:
                     self.debug.log(f"Failed to extract target coordinates: {e}")

        current_target = [target_x, target_y]
        self.debug.log(f"Starting drag target at: ({target_x:.2%}, {target_y:.2%})")

        # 3. Iterative refinement with visual feedback
        history: List[dict] = []

        work_ext = os.path.splitext(image_path)[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=work_ext, delete=False) as tf:
            work_path = tf.name
        self._temp_files.append(work_path)


        try:
            for i in range(max_iterations):
                shutil.copy2(image_path, work_path)

                # Convert to pixels for overlay
                # Target bbox (for visual box, not segmentation)
                target_px = [current_target[0] * img_w, current_target[1] * img_h]
                box_w = source_bbox_px[2] - source_bbox_px[0]
                box_h = source_bbox_px[3] - source_bbox_px[1]
                
                target_bbox_px = [
                    target_px[0] - box_w/2,
                    target_px[1] - box_h/2,
                    target_px[0] + box_w/2,
                    target_px[1] + box_h/2,
                ]

                # Draw overlay (with crop and paste)
                add_drag_overlay(
                    work_path,
                    source_bbox_px,
                    target_bbox=target_bbox_px,
                    target_center=tuple(target_px),
                    foreground_image=foreground_image,
                )

                # Save step image to debug
                self.debug.save_image(work_path, f"04_drag_step_{i+1}.png")

                # Ask for refinement with history
                result = self.planner.refine_drag(
                    work_path,
                    instruction,
                    current_target,
                    history,
                    source_description=source_desc,
                    target_description=target_description or "matching slot",
                    primary_goal=primary_goal or f"Drag {source_desc} to {target_description}",
                )

                conclusion = result.get("conclusion", "")
                decision = result.get("decision", "accept")
                dx = result.get("dx", 0)
                dy = result.get("dy", 0)

                # Clamp adjustment to avoid wild jumps
                max_step = 0.05
                if abs(dx) > max_step:
                    dx = math.copysign(max_step, dx)
                if abs(dy) > max_step:
                    dy = math.copysign(max_step, dy)

                self.debug.log(f"Iteration {i + 1}: '{conclusion}' -> {decision} (dx={dx:+.1%}, dy={dy:+.1%})")

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
                    self.debug.log("Drag target accepted")
                    break

                # Apply adjustment (relative)
                current_target[0] = max(0.0, min(1.0, current_target[0] + dx))
                current_target[1] = max(0.0, min(1.0, current_target[1] + dy))

            # After refinement loop, save final overlay image
            if os.path.exists(work_path):
                self.debug.save_image(work_path, "05_drag_final.png")

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
    """
    solver = CaptchaSolver(**kwargs)
    return solver.solve(image_path, instruction)
