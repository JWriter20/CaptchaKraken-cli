import sys
import os
# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.attention import AttentionExtractor

def get_hit_cells(bbox, rows=4, cols=4):
    # bbox is [x_min, y_min, x_max, y_max] normalized
    x1, y1, x2, y2 = bbox
    
    hit_cells = []
    for r in range(rows):
        for c in range(cols):
            # Cell bounds
            cx1 = c / cols
            cy1 = r / rows
            cx2 = (c + 1) / cols
            cy2 = (r + 1) / rows
            
            # Intersection
            ix1 = max(x1, cx1)
            iy1 = max(y1, cy1)
            ix2 = min(x2, cx2)
            iy2 = min(y2, cy2)
            
            if ix2 > ix1 and iy2 > iy1:
                # Calculate intersection area relative to cell area
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                cell_area = (cx2 - cx1) * (cy2 - cy1)
                ratio = inter_area / cell_area
                
                cell_num = r * cols + c + 1
                hit_cells.append((cell_num, ratio))
    return hit_cells

def main():
    image_path = "captchaimages/stairsCaptchaImage.png"
    
    print(f"Loading AttentionExtractor...")
    extractor = AttentionExtractor()
    
    prompts = ["stairs", "railing", "tree"]
    
    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        print(f"Running detect_objects('{prompt}') on {image_path}...")
        detections = extractor.detect_objects(image_path, prompt)
        
        print(f"Found {len(detections)} detections:")
        for i, det in enumerate(detections):
            bbox = det['bbox']
            score = det.get('score', 0.0)
            hits = get_hit_cells(bbox)
            hit_str = ", ".join([f"#{c}({r:.1%})" for c, r in hits])
            print(f"  {i+1}. Score: {score:.2f}, Hits: {hit_str}")
            print(f"     BBox: {bbox}")
            
        output_path = f"debug_{prompt}_dino.png"
        print(f"Visualizing to {output_path}...")
        extractor.visualize_detections(image_path, detections, output_path=output_path)

if __name__ == "__main__":
    main()


