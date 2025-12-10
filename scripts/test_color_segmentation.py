
import sys
import os
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from attention import AttentionExtractor

def test_color_seg():
    extractor = AttentionExtractor()
    image_path = "captchaimages/sourceWImage.png"
    
    print(f"Testing color segmentation on {image_path}...")
    
    # Test with k=5, disjoint components
    result = extractor.segment_by_color(image_path, k=5, merge_components=False)
    masks = result.get("masks", [])
    
    if masks:
        output_path = "debug_w_color_k5.png"
        extractor.visualize_masks(image_path, masks, output_path=output_path, draw_ids=True)
    
    # Test with k=3, merged components
    result = extractor.segment_by_color(image_path, k=3, merge_components=True)
    masks = result.get("masks", [])
    
    if masks:
        output_path = "debug_w_color_k3_merged.png"
        extractor.visualize_masks(image_path, masks, output_path=output_path, draw_ids=True)

if __name__ == "__main__":
    test_color_seg()

