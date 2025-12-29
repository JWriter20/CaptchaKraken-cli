import os
import tempfile
import shutil
from typing import List, Dict, Any, Tuple
from ..overlay import add_overlays_to_image

def segment(image_processor, attention_extractor, media_path: str, debug_manager=None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Pre-process, segment, and create numbered overlay for the image or video.
    Returns (path to labeled image, list of labeled objects).
    """
    ext = os.path.splitext(media_path)[1] or ".png"
    
    # 1. Ensure we have a static frame for processing if media is video
    is_video = any(media_path.lower().endswith(ext) for ext in [".mp4", ".webm", ".gif", ".avi"])
    
    if is_video:
        import cv2
        cap = cv2.VideoCapture(media_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read video from {media_path}")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            frame_path = tf.name
        cv2.imwrite(frame_path, frame)
        current_media = frame_path
    else:
        current_media = media_path
        frame_path = None

    # 2. Pre-process: Merge similar colors
    with tempfile.NamedTemporaryFile(suffix=f"_merged.png", delete=False) as tf:
        merged_path = tf.name
    
    sharpen_input = current_media
    try:
        image_processor.merge_similar_colors(current_media, merged_path, k=12)
        sharpen_input = merged_path
    except Exception as e:
        if debug_manager: debug_manager.log(f"Color merging failed: {e}")

    # 3. Greyscale image
    with tempfile.NamedTemporaryFile(suffix=f"_greyscale.png", delete=False) as tf:
        greyscale_path = tf.name
    try:
        image_processor.to_greyscale(sharpen_input, greyscale_path)
        sharpen_input = greyscale_path
    except Exception as e:
        if debug_manager: debug_manager.log(f"Greyscale conversion failed: {e}")

    # 4. Sharpen image
    with tempfile.NamedTemporaryFile(suffix=f"_sharp.png", delete=False) as tf:
        sharp_path = tf.name
    try:
        image_processor.sharpen_image(sharpen_input, sharp_path)
    except Exception as e:
        if debug_manager: debug_manager.log(f"Sharpening failed: {e}")
        shutil.copy2(sharpen_input, sharp_path)

    # 5. Segment with SAM 3
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
        if frame_path and os.path.exists(frame_path): os.unlink(frame_path)
        return current_media, []

    # 6. Create overlays
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

    # 7. Save overlay image
    with tempfile.NamedTemporaryFile(suffix=f"_labeled.png", delete=False) as tf:
        labeled_path = tf.name
    
    add_overlays_to_image(current_media, overlays, output_path=labeled_path)
    
    # Cleanup temp files
    for p in [merged_path, greyscale_path, sharp_path, frame_path]:
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except:
                pass
            
    return labeled_path, labeled_objects

