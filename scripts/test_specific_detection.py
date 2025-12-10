import argparse
import sys
import os
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.attention import AttentionExtractor
from src.overlay import add_overlays_to_image

def main():
    parser = argparse.ArgumentParser(description="Test detection and cropping")
    parser.add_argument("image_path", help="Path to image")
    parser.add_argument("prompt", help="Text description to detect")
    parser.add_argument("--max-area", type=float, default=0.2, help="Maximum allowed area ratio (0.0-1.0)")
    parser.add_argument("--box-thresh", type=float, default=None, help="Box threshold (default: auto)")
    parser.add_argument("--text-thresh", type=float, default=None, help="Text threshold (default: auto)")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: {args.image_path} not found")
        return

    print(f"Loading model and detecting '{args.prompt}' in {args.image_path}...")
    attention = AttentionExtractor()
    
    # Use robust detection
    prompts = [args.prompt]
    # Add some variations automatically
    if "letter" in args.prompt:
         prompts.append(args.prompt.replace("letter", "character"))
         prompts.append(args.prompt.replace("letter", "text"))
         prompts.append("white text")
    
    detections = attention.detect_robust(
        args.image_path, 
        prompts,
        min_score=args.box_thresh if args.box_thresh else 0.15,
        max_area=args.max_area
    )

    if not detections:
        print("No objects found.")
        return

    print(f"Found {len(detections)} raw objects. Filtering by max area {args.max_area}...")
    
    with Image.open(args.image_path) as img:
        img_w, img_h = img.size
        
        overlays = []
        valid_count = 0
        
        for i, det in enumerate(detections):
            bbox = det["bbox"] # [x_min, y_min, x_max, y_max] normalized
            
            # Check area
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]
            area = box_w * box_h
            
            if area > args.max_area:
                print(f"Skipping Object {i+1}: Score={det.get('score', 0):.2f}, Area={area:.2f} > {args.max_area}")
                continue
            
            # Aspect ratio check
            if box_w * 2.5 < box_h or box_h * 2.5 < box_w:
                print(f"Skipping Object {i+1}: Extreme aspect ratio")
                continue

            valid_count += 1
            
            # Convert to pixels
            x1 = bbox[0] * img_w
            y1 = bbox[1] * img_h
            x2 = bbox[2] * img_w
            y2 = bbox[3] * img_h
            
            w = x2 - x1
            h = y2 - y1
            
            print(f"Object {i+1} (VALID): Score={det.get('score', 0):.2f}, Area={area:.2f}, Box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            
            # Crop
            try:
                crop = img.crop((x1, y1, x2, y2))
                crop_name = f"crop_{valid_count}.png"
                crop.save(crop_name)
                print(f"Saved crop to {crop_name}")
            except Exception as e:
                print(f"Failed to crop: {e}")

            overlays.append({
                "bbox": [x1, y1, w, h],
                "text": f"{valid_count}",
                "color": "#00FF00"
            })

    if valid_count == 0:
        print("No objects found within area limits.")
        return

    # Save overlay
    add_overlays_to_image(args.image_path, overlays, "detection_result.png")
    print("Saved visualization to detection_result.png")

if __name__ == "__main__":
    main()

