import os
import tempfile
import shutil
import math
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Union
from ..overlay import add_drag_overlay

def dragPuzzlePredicate(detection):
    """
    Filters and weights detection priority for drag puzzles.
    Returns a priority score (float) or None if filtered out.
    Matches the exact logic from drag_labeler.py.
    """
    x1, y1, x2, y2 = detection["x_min"], detection["y_min"], detection["x_max"], detection["y_max"]
    width = x2 - x1
    height = y2 - y1
    score = detection.get("score", 0.0)

    # 1. Filter: Size limit (max 25% in either dimension)
    if width > 0.25 or height > 0.25:
        return None
    
    # 2. Priority: Boost items entirely in the right 20% of the image
    # (These are typically the source pieces to be moved)
    if x1 >= 0.70:
        score += 10.0 # Significant boost to ensure it wins over center/left items
        
    return score

def simulate_drag(
    solver,
    media_path: str,
    instruction: str,
    source_description: str,
    primary_goal: str,
    max_iterations: int = 5,
    source_bbox_override: Optional[List[float]] = None,
    current_location: Optional[List[float]] = None,
    mask_points_override: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """
    Solve drag puzzle with iterative refinement.
    Supports both images and videos.
    Returns a DragAction-compatible dictionary.
    """
    source_desc = source_description or "movable item"
    img_w, img_h = solver._image_size
    attention = solver._get_attention()
    mask_points = mask_points_override

    # 1. Find source
    if source_bbox_override:
        source_bbox_px = source_bbox_override
        source_x = (source_bbox_px[0] + source_bbox_px[2]) / 2 / img_w
        source_y = (source_bbox_px[1] + source_bbox_px[3]) / 2 / img_h
    else:
        # Check if source_desc is an Object ID (e.g. "Object 4")
        import re
        obj_match = re.search(r"Object\s*(\d+)", str(source_desc), re.IGNORECASE)
        found_obj = None
        if obj_match:
            obj_id = int(obj_match.group(1))
            # solver might have current_objects if called from _solve_general
            objects = getattr(solver, "current_objects", [])
            for o in objects:
                if o.get("id") == obj_id:
                    found_obj = o
                    break
        
        if found_obj:
            b = found_obj["bbox"] # [x, y, w, h]
            source_bbox_px = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
            source_x = (b[0] + b[2]/2) / img_w
            source_y = (b[1] + b[3]/2) / img_h
            if "mask_points" in found_obj:
                mask_points = found_obj["mask_points"]
        else:
            # Use the same detection + predicate logic as drag_labeler.py
            # Request more detections to allow for filtering
            objs_to_detect = 10
            detections = attention.detect(media_path, str(source_desc), max_objects=objs_to_detect)
            print(f"Detections: {detections}")
            if detections:
                # Apply predicate and filter
                scored_detections = []
                for d in detections:
                    priority = dragPuzzlePredicate(d)
                    if priority is not None:
                        d["priority"] = priority
                        scored_detections.append(d)
                
                if scored_detections:
                    # Sort by priority descending
                    scored_detections.sort(key=lambda x: x["priority"], reverse=True)
                    # Take the best detection
                    obj = scored_detections[0]
                    source_x = (obj["x_min"] + obj["x_max"]) / 2
                    source_y = (obj["y_min"] + obj["y_max"]) / 2
                    if solver.debug: solver.debug.log(f"Detected source '{source_desc}' at center ({source_x:.3f}, {source_y:.3f}) with priority {obj.get('priority', 0.0):.2f}")
                    source_bbox_px = [
                        obj["x_min"] * img_w,
                        obj["y_min"] * img_h,
                        obj["x_max"] * img_w,
                        obj["y_max"] * img_h
                    ]
                    # Some detectors might provide mask points
                    if "mask_points" in obj:
                        mask_points = obj["mask_points"]
                else:
                    # All detections filtered out, use fallback
                    if solver.debug: solver.debug.log(f"All detections for '{source_desc}' were filtered out by dragPuzzlePredicate, using fallback")
                    source_x, source_y = 0.5, 0.5
                    box_size = 0.1
                    source_bbox_px = [
                        (source_x - box_size/2) * img_w,
                        (source_y - box_size/2) * img_h,
                        (source_x + box_size/2) * img_w,
                        (source_y + box_size/2) * img_h
                    ]
            else:
                source_x, source_y = 0.5, 0.5
                box_size = 0.1
                source_bbox_px = [
                    (source_x - box_size/2) * img_w,
                    (source_y - box_size/2) * img_h,
                    (source_x + box_size/2) * img_w,
                    (source_y + box_size/2) * img_h
                ]


    # 2. Initial target estimate
    if current_location and len(current_location) >= 2:
        target_x, target_y = current_location[0], current_location[1]
    else:
        target_x, target_y = 0.5, 0.5

    current_target = [target_x, target_y]

    # 4. Prepare foreground image (optional but helpful for visual refinement)
    foreground_image = None
    try:
        x1, y1, x2, y2 = map(int, source_bbox_px)
        is_video = any(media_path.lower().endswith(ext) for ext in [".mp4", ".webm", ".gif", ".avi"])
        
        if is_video:
            import cv2
            cap = cv2.VideoCapture(media_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                img = None
        else:
            img = Image.open(media_path)

        if img:
            w, h = img.size
            if mask_points:
                # Create precise foreground from mask points
                mask = Image.new("L", (w, h), 0)
                from PIL import ImageDraw
                draw_mask = ImageDraw.Draw(mask)
                pts = [(p[0] * w, p[1] * h) for p in mask_points]
                draw_mask.polygon(pts, fill=255)
                
                rgba_img = img.convert("RGBA")
                rgba_img.putalpha(mask)
                foreground_image = rgba_img.crop((x1, y1, x2, y2))
            else:
                # Add 10% padding to the crop for better context during background removal
                bw = x2 - x1
                bh = y2 - y1
                pad_x = int(bw * 0.1)
                pad_y = int(bh * 0.1)
                
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w, x2 + pad_x)
                cy2 = min(h, y2 + pad_y)
                
                if cx2 > cx1 and cy2 > cy1:
                    source_crop = img.crop((cx1, cy1, cx2, cy2))
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                        crop_path = tf.name
                    source_crop.save(crop_path)
                    
                    # Try background removal on the padded crop
                    foreground_image = solver.image_processor.remove_background(
                        crop_path, 
                        prompt=source_desc,
                        max_objects=1
                    )
                    
                    # If foreground removal succeeded, we need to crop the padding back out
                    # or ensure it matches the original requested bbox size.
                    if foreground_image:
                        # The foreground image is RGBA. We want to return just the object part.
                        # Actually, remove_background already does that.
                        pass
                    
                    os.unlink(crop_path)
    except Exception as e:
        if solver.debug: solver.debug.log(f"Failed to prepare foreground image: {e}")

    history: List[dict] = []

    # Get a static frame for overlaying progress
    if is_video:
        import cv2
        cap = cv2.VideoCapture(media_path)
        ret, frame = cap.read()
        cap.release()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            frame_path = tf.name
        cv2.imwrite(frame_path, frame)
    else:
        frame_path = media_path

    ext = os.path.splitext(frame_path)[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
        work_path = tf.name

    target_bbox_px = [0, 0, 0, 0] # Initialize
    try:
        for i in range(max_iterations):
            shutil.copy2(frame_path, work_path)
            target_px = [current_target[0] * img_w, current_target[1] * img_h]
            box_w = source_bbox_px[2] - source_bbox_px[0]
            box_h = source_bbox_px[3] - source_bbox_px[1]
            
            target_bbox_px = [
                target_px[0] - box_w/2,
                target_px[1] - box_h/2,
                target_px[0] + box_w/2,
                target_px[1] + box_h/2,
            ]

            add_drag_overlay(
                work_path,
                source_bbox_px,
                target_bbox=target_bbox_px,
                target_center=tuple(target_px),
                foreground_image=foreground_image,
                mask_points=mask_points,
            )

            if solver.debug:
                solver.debug.save_image(work_path, f"04_drag_step_{i+1}.png")

            result = solver.planner.refine_drag(
                work_path,
                instruction,
                current_target,
                history,
                source_description=source_desc,
                target_description=primary_goal, # Use the clean target description
                primary_goal=primary_goal,
                iteration=i,
            )

            decision = result.get("decision", "accept")
            dx, dy = result.get("dx", 0.0), result.get("dy", 0.0)

            # Adaptive max_step: Allow large initial move, then force refinement
            if i == 0:
                max_step = 0.5  # Up to 50% for initial estimation
            else:
                # Decrease max_step as we iterate to force smaller refinements
                max_step = max(0.02, 0.15 / (i + 1))
            
            dx = math.copysign(min(abs(dx), max_step), dx)
            dy = math.copysign(min(abs(dy), max_step), dy)

            history.append({
                "destination": current_target.copy(),
                "decision": decision,
            })

            if decision == "accept" or (abs(dx) < 0.005 and abs(dy) < 0.005):
                break

            current_target[0] = max(0.0, min(1.0, current_target[0] + dx))
            current_target[1] = max(0.0, min(1.0, current_target[1] + dy))

    finally:
        if os.path.exists(work_path):
            os.unlink(work_path)
        if is_video and os.path.exists(frame_path):
            os.unlink(frame_path)

    # Final conversion to percentages for return
    source_bbox_pct = [
        source_bbox_px[0] / img_w,
        source_bbox_px[1] / img_h,
        source_bbox_px[2] / img_w,
        source_bbox_px[3] / img_h
    ]
    
    # For target, we use the last calculated target_bbox_px
    target_bbox_pct = [
        target_bbox_px[0] / img_w,
        target_bbox_px[1] / img_h,
        target_bbox_px[2] / img_w,
        target_bbox_px[3] / img_h
    ]

    return {
        "action": "drag",
        "source_bounding_box": source_bbox_pct,
        "target_bounding_box": target_bbox_pct,
    }


