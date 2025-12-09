import os
import sys
import numpy as np
import shutil
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.imagePreprocessing import apply_gaussian_blur, apply_clahe
from src.attention import AttentionExtractor
from src.overlay import add_overlays_to_image

def test_dino_sam2_hybrid():
    image_path = "captchaimages/hcaptchaDragImage4.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    print(f"Processing {image_path} with DINO + SAM2 Hybrid...")

    # Optional: Preprocessing
    # Sharpen image before DINO detection to help find letters
    sharpened_path = "test-results/hcaptchaDragImage4_sharpened_dino.png"
    if not os.path.exists(sharpened_path):
        print("Sharpening image for DINO...")
        # Use existing sharpen_image from imagePreprocessing
        from src.imagePreprocessing import sharpen_image
        sharpen_image(image_path, sharpened_path)
    
    # Use sharpened image for detection
    detect_path = sharpened_path
    
    attention = AttentionExtractor()
    
    # 1. Detect with GroundingDINO
    # Prompt: "letter" or "character" or "shape"
    # We lower the threshold to increase sensitivity as requested.
    prompt = "letter . character"
    box_threshold = 0.25 # Lower than default 0.35
    text_threshold = 0.25
    
    print(f"Running GroundingDINO with prompt='{prompt}', box_thresh={box_threshold}...")
    dino_result = attention.detect_with_grounding_dino(
        detect_path, 
        prompt, 
        box_threshold=box_threshold, 
        text_threshold=text_threshold
    )
    
    objects = dino_result.get("objects", [])
    print(f"DINO found {len(objects)} objects")
    
    if not objects:
        print("No objects found by DINO. Exiting.")
        return

    # Convert DINO objects to SAM2 input boxes
    # DINO returns normalized [x_min, y_min, x_max, y_max]
    input_boxes = []
    for obj in objects:
        # Filter out very low confidence if we went too low?
        # For now, trust the threshold.
        bbox = [obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]]
        input_boxes.append(bbox)
        print(f"  - {obj['label']} ({obj['score']:.2f}): {bbox}")

    # 2. Segment with SAM2 using DINO boxes
    print("\nRunning SAM2 with DINO boxes as prompts...")
    # We use the SHARPENED image for segmentation too, as edges will be clearer
    sam_result = attention.segment_with_sam2(
        detect_path,
        input_boxes=input_boxes,
        score_threshold=0.5 # Basic check to ensure mask matches box well
    )
    
    masks = sam_result.get("masks", [])
    print(f"Got {len(masks)} masks from SAM2")

    # 3. Filter and process results
    boxes = []
    with Image.open(detect_path) as img:
        img_w, img_h = img.size
    
    total_area = img_w * img_h
    
    for i, mask in enumerate(masks):
        if not isinstance(mask, np.ndarray):
            continue
            
        # Basic sanity checks
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue
            
        x1, x2 = np.min(x_indices), np.max(x_indices)
        y1, y2 = np.min(y_indices), np.max(y_indices)
        w = x2 - x1
        h = y2 - y1
        
        # Area check
        area = w * h
        mask_area = np.sum(mask)
        solidity = mask_area / area if area > 0 else 0
        
        # DINO might find the whole sidebar as a "letter" sometimes, or tiny noise.
        # Filter tiny things
        if w < 10 or h < 10:
            continue
            
        # Filter massive things (background)
        if area > (total_area * 0.4):
             print(f"  - Dropping mask {i} (too large)")
             continue
             
        # Filter hollow things
        if solidity < 0.15:
            print(f"  - Dropping mask {i} (low solidity: {solidity:.2f})")
            continue

        boxes.append({
            "x1": int(x1), "y1": int(y1), 
            "x2": int(x2), "y2": int(y2),
            "w": int(w), "h": int(h),
            "area": int(area),
            "solidity": float(solidity),
            "origin": "dino_sam2"
        })

    # Filter overlapping (IoU) - DINO might return multiple boxes for same object
    boxes.sort(key=lambda b: b["area"], reverse=True)
    keep = []
    
    def compute_iou(b1, b2):
        xx1 = max(b1["x1"], b2["x1"])
        yy1 = max(b1["y1"], b2["y1"])
        xx2 = min(b1["x2"], b2["x2"])
        yy2 = min(b1["y2"], b2["y2"])
        
        w_inter = max(0, xx2 - xx1)
        h_inter = max(0, yy2 - yy1)
        inter = w_inter * h_inter
        
        union = b1["area"] + b2["area"] - inter
        return inter / union if union > 0 else 0

    for b in boxes:
        is_overlap = False
        for k in keep:
            if compute_iou(b, k) > 0.6: 
                is_overlap = True
                break
        if not is_overlap:
            keep.append(b)

    print(f"Kept {len(keep)} boxes after filtering")

    # 4. Create overlays
    overlays = []
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F1C40F", "#9B59B6", "#E67E22"]
    
    for i, b in enumerate(keep):
        overlays.append({
            "bbox": [b["x1"], b["y1"], b["w"], b["h"]], # [x, y, w, h]
            "number": i + 1,
            "color": colors[i % len(colors)]
        })

    # 5. Save labeled image
    labeled_path = "test-results/hcaptchaDragImage4_segmented_DINO_SAM2.png"
    os.makedirs("test-results", exist_ok=True)
    shutil.copy2(image_path, labeled_path)
    
    add_overlays_to_image(labeled_path, overlays)
    print(f"Saved segmented overlay to {labeled_path}")

if __name__ == "__main__":
    test_dino_sam2_hybrid()

