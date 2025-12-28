import os
import tempfile
import shutil
from typing import List, Dict, Any, Tuple
from ..overlay import add_overlays_to_image

def segment(image_processor, attention_extractor, image_path: str, debug_manager=None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Pre-process, segment, and create numbered overlay for the image.
    Returns (path to labeled image, list of labeled objects).
    """
    ext = os.path.splitext(image_path)[1] or ".png"
    
    # 1. Pre-process: Merge similar colors
    with tempfile.NamedTemporaryFile(suffix=f"_merged{ext}", delete=False) as tf:
        merged_path = tf.name
    
    sharpen_input = image_path
    try:
        image_processor.merge_similar_colors(image_path, merged_path, k=12)
        sharpen_input = merged_path
    except Exception as e:
        if debug_manager: debug_manager.log(f"Color merging failed: {e}")

    # 2. Greyscale image (optional step from solver.py)
    with tempfile.NamedTemporaryFile(suffix=f"_greyscale{ext}", delete=False) as tf:
        greyscale_path = tf.name
    try:
        image_processor.to_greyscale(sharpen_input, greyscale_path)
        sharpen_input = greyscale_path
    except Exception as e:
        if debug_manager: debug_manager.log(f"Greyscale conversion failed: {e}")

    # 3. Sharpen image
    with tempfile.NamedTemporaryFile(suffix=f"_sharp{ext}", delete=False) as tf:
        sharp_path = tf.name
    try:
        image_processor.sharpen_image(sharpen_input, sharp_path)
    except Exception as e:
        if debug_manager: debug_manager.log(f"Sharpening failed: {e}")
        shutil.copy2(sharpen_input, sharp_path)

    # 4. Segment with SAM 3
    # Use a generic prompt to find interactable elements
    from PIL import Image
    detections = attention_extractor.detect(sharp_path, "interactable button or checkbox", max_objects=20)
    
    with Image.open(sharp_path) as img:
        img_w, img_h = img.size
        
    objects = []
    for det in detections:
        x1, y1, x2, y2 = det["x_min"] * img_w, det["y_min"] * img_h, det["x_max"] * img_w, det["y_max"] * img_h
        objects.append({
            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "score": det["score"]
        })
    
    if not objects:
        return image_path, []

    # 5. Create overlays
    overlays = []
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F1C40F", "#9B59B6", "#E67E22"]
    labeled_objects = []
    
    for i, obj in enumerate(objects):
        bbox = obj["bbox"] # [x, y, w, h]
        overlays.append({
            "bbox": bbox,
            "number": i + 1,
            "color": colors[i % len(colors)]
        })
        obj_with_id = obj.copy()
        obj_with_id["id"] = i + 1
        labeled_objects.append(obj_with_id)

    # 6. Save overlay image
    with tempfile.NamedTemporaryFile(suffix=f"_labeled{ext}", delete=False) as tf:
        labeled_path = tf.name
    
    add_overlays_to_image(image_path, overlays, output_path=labeled_path)
    
    # Cleanup temp files
    for p in [merged_path, greyscale_path, sharp_path]:
        if os.path.exists(p):
            os.unlink(p)
            
    return labeled_path, labeled_objects

