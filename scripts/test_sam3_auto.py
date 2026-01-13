import os
import sys
import numpy as np
from PIL import Image
import torch
from transformers import pipeline

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from attention import AttentionExtractor, clusterDetections, default_predicate

def test_sam3_auto():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        generator = pipeline("mask-generation", model="facebook/sam3", device=device, trust_remote_code=True)
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return

    test_images = [
        ("captchaimages/hcaptchaChooseSimilarShapes.png", 5),
        ("captchaimages/hcaptchaDragImage1.png", (2, 7)),
        ("captchaimages/hcaptchaDragImages3.png", (4, 6)),
        ("captchaimages/hcaptchaDragImage4.png", (4, 6)),
        ("captchaimages/hcaptchaPuzzle.png", (6, 8)),
        ("captchaimages/hcaptchaPuzzle2.png", 5),
    ]
    
    output_dir = "sam3_auto_test"
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = AttentionExtractor() # Use for visualization
    results_summary = {}

    for img_path, expected in test_images:
        if not os.path.exists(img_path):
            print(f"Skipping {img_path}, not found.")
            continue
            
        print(f"\nTesting {img_path} (Expected: {expected})")
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        
        print("  Generating masks...", end="", flush=True)
        try:
            results = generator(image)
            masks = results["masks"]
            print(f" Found {len(masks)} masks.")
            
            detections = []
            for mask in masks:
                # Ensure mask is a numpy array
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                
                # If it's (1, H, W) or (H, W)
                if len(mask.shape) == 3:
                    mask = mask[0]
                
                # Find bounding box of mask
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if not np.any(rows) or not np.any(cols):
                    continue
                
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                det = {
                    "x_min": float(cmin) / width,
                    "y_min": float(rmin) / height,
                    "x_max": float(cmax) / width,
                    "y_max": float(rmax) / height,
                    "score": 0.5,
                }
                
                # Apply 15% y-filter as in prompt method
                y_center = (det["y_min"] + det["y_max"]) / 2
                if y_center < 0.15 or y_center > 0.85:
                    continue
                
                score = default_predicate(det)
                if score is not None:
                    det["score"] = score
                    det["bbox"] = [det["x_min"], det["y_min"], det["x_max"], det["y_max"]]
                    detections.append(det)
            
            # Merge overlapping detections
            merged = clusterDetections(detections, iou_threshold=0.4, distance_threshold=0.1)
            count = len(merged)
            results_summary[img_path] = count
            
            # Check if within expected range
            if isinstance(expected, tuple):
                passed = expected[0] <= count <= expected[1]
            else:
                passed = count == expected
            
            status = "PASS" if passed else "FAIL"
            print(f"  After filtering/clustering: {count} objects. [{status}]")
            
            # Visualize
            img_name = os.path.basename(img_path).replace(".png", "")
            output_path = os.path.join(output_dir, f"{img_name}_auto.png")
            extractor.visualize_detections(img_path, merged, output_path=output_path)
            
        except Exception as e:
            print(f" Error: {e}")
            results_summary[img_path] = str(e)

    print("\n" + "="*50)
    print("SUMMARY (SAM3 Auto Mask Generation)")
    print("="*50)
    for img_path, count in results_summary.items():
        print(f"{img_path}: {count}")

if __name__ == "__main__":
    test_sam3_auto()

