import os
import sys
import numpy as np
import shutil
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.attention import AttentionExtractor
from src.overlay import add_overlays_to_image

def test_sam2_standalone():
    image_path = "captchaimages/hcaptchaDragImage2.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    print(f"Processing {image_path} with SAM2 (Grid Mode)...")

    attention = AttentionExtractor()
    
    # Run SAM2 with automatic grid prompts
    print("Running SAM2 with detectInteractable (includes filtering)...")
    
    # Use the new filtered detection method
    result = attention.detectInteractable(
        image_path,
        points_per_side=8, 
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
    output_path = "test-results/hcaptchaDragImage2_SAM2_standalone.png"
    os.makedirs("test-results", exist_ok=True)
    shutil.copy2(image_path, output_path)
    
    add_overlays_to_image(output_path, overlays)
    
    print(f"Saved segmented overlay to {output_path}")


if __name__ == "__main__":
    test_sam2_standalone()

