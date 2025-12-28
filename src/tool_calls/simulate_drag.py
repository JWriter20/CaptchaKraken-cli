import os
import tempfile
import shutil
import math
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from ..overlay import add_drag_overlay

def simulate_drag(
    solver,
    image_path: str,
    instruction: str,
    source_description: str,
    target_description: str,
    max_iterations: int = 5,
    source_bbox_override: Optional[List[float]] = None,
    initial_location_hint: Optional[List[float]] = None,
    primary_goal: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Solve drag puzzle with iterative refinement.
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
        detections = attention.detect(image_path, source_desc, max_objects=1)
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

    source_coords = [source_x, source_y]

    # 2. Prepare foreground image
    foreground_image = None
    try:
        x1, y1, x2, y2 = map(int, source_bbox_px)
        with Image.open(image_path) as img:
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

    # 3. Initial target estimate
    target_x, target_y = 0.5, 0.5
    if initial_location_hint:
        target_x, target_y = initial_location_hint
    elif target_description and target_description != "matching slot":
        detections = attention.detect(image_path, target_description, max_objects=1)
        if detections:
            obj = detections[0]
            target_x, target_y = (obj["x_min"] + obj["x_max"]) / 2, (obj["y_min"] + obj["y_max"]) / 2
        else:
            target_x, target_y = 0.5, 0.5

    current_target = [target_x, target_y]
    history: List[dict] = []

    ext = os.path.splitext(image_path)[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
        work_path = tf.name

    try:
        for i in range(max_iterations):
            shutil.copy2(image_path, work_path)
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
                target_description=target_description or "matching slot",
                primary_goal=primary_goal or f"Drag {source_desc} to {target_description}",
            )

            decision = result.get("decision", "accept")
            dx, dy = result.get("dx", 0.0), result.get("dy", 0.0)

            max_step = 0.05
            dx = math.copysign(min(abs(dx), max_step), dx)
            dy = math.copysign(min(abs(dy), max_step), dy)

            history.append({
                "destination": current_target.copy(),
                "conclusion": result.get("conclusion", ""),
                "decision": decision,
            })

            if decision == "accept" or (abs(dx) < 0.005 and abs(dy) < 0.005):
                break

            current_target[0] = max(0.0, min(1.0, current_target[0] + dx))
            current_target[1] = max(0.0, min(1.0, current_target[1] + dy))

    finally:
        if os.path.exists(work_path):
            os.unlink(work_path)

    return {
        "action": "drag",
        "source_coordinates": source_coords,
        "target_coordinates": current_target,
    }

