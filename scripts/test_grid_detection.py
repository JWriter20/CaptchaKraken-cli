import sys
import os
# import cv2  # Not needed for visualization anymore

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.image_processor import ImageProcessor
from src.overlay import add_overlays_to_image

def test_image(image_path):
    print(f"Testing {image_path}...")
    if not os.path.exists(image_path):
        print(f"  File not found: {image_path}")
        return

    boxes = ImageProcessor.get_grid_bounding_boxes(image_path)
    
    if boxes:
        print(f"  Success: Found {len(boxes)} boxes")
        print(f"  Sample: {boxes[0]}")
        
        # Prepare overlays
        # Convert (x1, y1, x2, y2) to [x, y, w, h] for overlay utility
        overlays = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            w = x2 - x1
            h = y2 - y1
            overlays.append({
                'bbox': [x1, y1, w, h],
                'number': i + 1,
                'color': '#E74C3C'  # Standard bright red for high visibility and contrast
            })
            
        out_path = f"test-results/grid_det_{os.path.basename(image_path)}"
        os.makedirs("test-results", exist_ok=True)
        
        try:
            add_overlays_to_image(image_path, overlays, output_path=out_path)
            print(f"  Saved visualization to {out_path}")
        except Exception as e:
            print(f"  Error saving visualization: {e}")
    else:
        print("  No grid detected.")

if __name__ == "__main__":
    test_image("captchaimages/hcaptchaImages1.png")
    test_image("captchaimages/recaptchaImages2.png")
    test_image("captchaimages/recaptchaBasic.png")
