import os
import tempfile
import shutil
import math
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Union
from ..overlay import add_drag_overlay

def simulate_drag(
    solver,
    media_path: str,
    instruction: str,
    source_description: str,
    primary_goal: str,
    max_iterations: int = 5,
    source_bbox_override: Optional[List[float]] = None,
    location_hint: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Solve drag puzzle with iterative refinement.
    Supports both images and videos.
    Returns a DragAction-compatible dictionary.
    """
    source_desc = source_description or "movable item"
    img_w, img_h = solver._image_size
    attention = solver._get_attention()

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
        else:
            detections = attention.detect(media_path, str(source_desc), max_objects=1)
            if detections:
                obj = detections[0]
                source_x = (obj["x_min"] + obj["x_max"]) / 2
                source_y = (obj["y_min"] + obj["y_max"]) / 2
                source_bbox_px = [
                    obj["x_min"] * img_w,
                    obj["y_min"] * img_h,
                    obj["x_max"] * img_w,
                    obj["y_max"] * img_h
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
    if location_hint and len(location_hint) >= 2:
        target_x, target_y = location_hint[0], location_hint[1]
    else:
        target_x, target_y = 0.5, 0.5
    # Use primary_goal as the target_description
    t_desc = primary_goal
    if t_desc:
        # Check if t_desc is an Object ID
        import re
        obj_match = re.search(r"Object\s*(\d+)", str(t_desc), re.IGNORECASE)
        found_obj = None
        if obj_match:
            obj_id = int(obj_match.group(1))
            objects = getattr(solver, "current_objects", [])
            for o in objects:
                if o.get("id") == obj_id:
                    found_obj = o
                    break
        
        if found_obj:
            b = found_obj["bbox"]
            target_x = (b[0] + b[2]/2) / img_w
            target_y = (b[1] + b[3]/2) / img_h
        else:
            detections = attention.detect(media_path, str(t_desc), max_objects=1)
            if detections:
                obj = detections[0]
                target_x, target_y = (obj["x_min"] + obj["x_max"]) / 2, (obj["y_min"] + obj["y_max"]) / 2
            else:
                target_x, target_y = 0.5, 0.5
    else:
        target_x, target_y = 0.5, 0.5

    current_target = [target_x, target_y]

    # 3. Prepare foreground image (optional but helpful for visual refinement)
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
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 > x1 and y2 > y1:
                source_crop = img.crop((x1, y1, x2, y2))
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    crop_path = tf.name
                source_crop.save(crop_path)
                
                foreground_image = solver.image_processor.remove_background(
                    crop_path, 
                    prompt=source_desc,
                    max_objects=1
                )
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
            )

            decision = result.get("decision", "accept")
            dx, dy = result.get("dx", 0.0), result.get("dy", 0.0)

            max_step = 0.05
            dx = math.copysign(min(abs(dx), max_step), dx)
            dy = math.copysign(min(abs(dy), max_step), dy)

            history.append({
                "destination": current_target.copy(),
                "analysis": result.get("analysis", ""),
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


