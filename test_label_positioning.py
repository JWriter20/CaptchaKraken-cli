
import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from attention import AttentionExtractor
from imagePreprocessing import merge_similar_colors

def test_label_positioning():
    extractor = AttentionExtractor()
    image_path = "captchaimages/sourceWImage.png"
    
    print(f"Testing label positioning on {image_path}...")
    
    # Re-run the segmentation parameters that worked well for separating W and BG
    k_val = 3
    preprocessed_path = "temp_quantized_w_labels.png"
    merge_similar_colors(image_path, preprocessed_path, k=k_val)
    
    min_area = 0.01 
    
    result = extractor.segment_by_color(preprocessed_path, k=k_val, merge_components=False, min_area_ratio=min_area)
    masks = result.get("masks", [])
    colors = result.get("colors", [])
    
    print(f"Found {len(masks)} masks.")
    
    # Visualization
    if masks:
        output_path = "debug_w_labels_fixed.png"
        extractor.visualize_masks(image_path, masks, output_path=output_path, draw_ids=True, colors=colors)
        print(f"Result saved to {output_path}")

    # Clean up
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

if __name__ == "__main__":
    test_label_positioning()

