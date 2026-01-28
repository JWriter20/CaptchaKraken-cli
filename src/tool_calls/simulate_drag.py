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
    primary_goal: str,
    source_id: int,
    target_id: Optional[int] = None,
    max_iterations: int = 5,
) -> Dict[str, Any]:
    """
    Solve drag puzzle with iterative refinement.
    Supports both images and videos.
    Returns a DragAction-compatible dictionary.
    """
    img_w, img_h = solver._image_size
    
    # 1. Find source object by ID
    found_obj = None
    objects = getattr(solver, "current_objects", [])
    for o in objects:
        if o.get("id") == source_id:
            found_obj = o
            break
    
    if not found_obj:
        if solver.debug:
            solver.debug.log(f"Error: source_id {source_id} not found in current objects")
        # Return a dummy or fail? Let's return empty/done-like
        return {
            "action": "done",
            "source_bounding_box": [0, 0, 0, 0],
            "target_bounding_box": [0, 0, 0, 0],
        }

    b = found_obj["bbox"] # [x1, y1, x2, y2] normalized
    source_bbox_px = [b[0] * img_w, b[1] * img_h, b[2] * img_w, b[3] * img_h]
    source_x = (b[0] + b[2]) / 2
    source_y = (b[1] + b[3]) / 2
    mask_points = found_obj.get("mask_points")
    source_desc = found_obj.get("label", f"Object {source_id}")

    # 2. Find target location
    target_x, target_y = 0.5, 0.5 # Default fallback
    if target_id is not None:
        found_target = None
        for o in objects:
            if o.get("id") == target_id:
                found_target = o
                break
        
        if found_target:
            tb = found_target["bbox"]
            target_x = (tb[0] + tb[2]) / 2
            target_y = (tb[1] + tb[3]) / 2
            if solver.debug:
                solver.debug.log(f"Using target_id {target_id} at ({target_x:.3f}, {target_y:.3f})")

    current_target = [target_x, target_y]

    # 3. Prepare foreground image (cutout of the source object)
    foreground_image = None
    try:
        x1, y1, x2, y2 = map(int, source_bbox_px)
        is_video = any(media_path.lower().endswith(ext) for ext in [".mp4", ".gif", ".avi"])
        
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
                # Fallback to simple crop if no mask points
                foreground_image = img.crop((x1, y1, x2, y2)).convert("RGBA")
    except Exception as e:
        if solver.debug:
            solver.debug.log(f"Failed to prepare foreground image: {e}")

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

    target_bbox_px = [0, 0, 0, 0]
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
                target_description=primary_goal,
                primary_goal=primary_goal,
                iteration=i,
            )

            decision = result.get("decision", "accept")
            dx, dy = result.get("dx", 0.0), result.get("dy", 0.0)

            if i == 0:
                max_step = 0.5
            else:
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

    source_bbox_pct = [
        source_bbox_px[0] / img_w,
        source_bbox_px[1] / img_h,
        source_bbox_px[2] / img_w,
        source_bbox_px[3] / img_h
    ]
    
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


