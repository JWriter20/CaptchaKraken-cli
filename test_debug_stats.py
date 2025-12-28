import os
import sys
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.tool_calls.find_grid import find_grid

def test_slanted_grid_debug():
    image_path = "captchaimages/slantedGrid.png"
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Re-implementing the mask logic to inspect components
    from src.tool_calls.find_grid import _get_similar_color_segments
    h_mask = _get_similar_color_segments(lab, axis=1, min_len=300, max_len=1500)
    v_mask = _get_similar_color_segments(lab, axis=0, min_len=300, max_len=1500)
    
    a_chan = lab[:, :, 1].astype(np.int16)
    b_chan = lab[:, :, 2].astype(np.int16)
    sat_thresh = 10
    is_neutral = (np.abs(a_chan - 128) < sat_thresh) & (np.abs(b_chan - 128) < sat_thresh)
    neutral_mask = is_neutral.astype(np.uint8) * 255
    
    grid_mask = cv2.bitwise_or(h_mask, v_mask)
    grid_mask = cv2.bitwise_and(grid_mask, neutral_mask)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(grid_mask, connectivity=8)
    
    print(f"Found {num_labels} components total.")
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        print(f"Component {i}: x={x}, y={y}, w={w}, h={h}, area={area}")

if __name__ == "__main__":
    test_slanted_grid_debug()

