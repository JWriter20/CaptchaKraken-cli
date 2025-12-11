import os
import sys
import numpy as np
import shutil
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.attention import AttentionExtractor
from src.overlay import add_overlays_to_image

def test_sam2_standalone(image_path=None):
    if image_path is None:
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            image_path = "captchaimages/hcaptchaDragImage2.png"

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing {image_path} with SAM2 (Grid Mode)...")

    attention = AttentionExtractor()
    
    # --- DEBUG STEP: Run segment_with_sam2 directly to visualize grid and raw masks ---
    print("Running raw SAM2 segmentation for debugging...")
    sam_result = attention.segment_with_sam2(
        image_path,
        points_per_side=10, # 100 points
        score_threshold=0.0 # Get all masks
    )
    
    raw_masks = sam_result.get("masks", [])
    points = sam_result.get("points", [])
    
    print(f"Raw SAM2 returned {len(raw_masks)} masks.")
    
    # Visualize grid points
    if points is not None:
        import matplotlib.pyplot as plt
        
        with Image.open(image_path) as img:
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            
            # Points is (N, 2)
            pts_x = [p[0] for p in points]
            pts_y = [p[1] for p in points]
            
            plt.scatter(pts_x, pts_y, c='red', s=20, marker='x', label='Grid Points')
            plt.legend()
            plt.axis('off')
            
            grid_out = f"test-results/{base_name}_grid_points.png"
            plt.savefig(grid_out, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved grid points visualization to {grid_out}")

    # Visualize ALL raw masks
    if raw_masks:
        masks_out = f"test-results/{base_name}_all_raw_masks.png"
        attention.visualize_masks(image_path, raw_masks, output_path=masks_out)
        print(f"Saved all raw masks visualization to {masks_out}")
    
    # --------------------------------------------------------------------------------
    
    # Run SAM2 with automatic grid prompts
    print("Running SAM2 with detectInteractable (includes filtering)...")
    
    # Use the new filtered detection method
    result = attention.detectInteractable(
        image_path,
        points_per_side=10, 
        score_threshold=0.5,
        max_area_ratio=0.4,
        aspect_ratio_threshold=4.0
    )
    
    objects = result.get("objects", [])
    print(f"Got {len(objects)} objects from detectInteractable (after filtering)")

    # Create overlays from returned objects
    overlays = []
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F1C40F", "#9B59B6", "#E67E22"]
    
    for i, obj in enumerate(objects):
        bbox = obj["bbox"] # [x, y, w, h]
        overlays.append({
            "bbox": bbox,
            "number": i + 1,
            "color": colors[i % len(colors)]
        })

    # Save labeled image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"test-results/{base_name}_SAM2_standalone.png"
    os.makedirs("test-results", exist_ok=True)
    shutil.copy2(image_path, output_path)
    
    add_overlays_to_image(output_path, overlays)
    
    print(f"Saved segmented overlay to {output_path}")


if __name__ == "__main__":
    test_sam2_standalone()

