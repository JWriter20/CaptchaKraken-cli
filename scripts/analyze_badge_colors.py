#!/usr/bin/env python3
"""Analyze HSV values of blue badge in selected cells"""
import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.image_processor import ImageProcessor

# Test image
image_path = os.path.join(PROJECT_ROOT, "captchaimages", "coreRecaptcha", "motorcycle3of7Selected.png")

print(f"Analyzing: {image_path}")

# Initialize processor
processor = ImageProcessor()

# Detect grid
grid_boxes = ImageProcessor.get_grid_bounding_boxes(image_path)
print(f"Found {len(grid_boxes)} grid cells\n")

# Expected selected cells
expected = [6, 10, 11]

img = cv2.imread(image_path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
bgr = img.copy()

for cell_id in expected:
    idx = cell_id - 1
    if idx < 0 or idx >= len(grid_boxes):
        continue
    
    x1, y1, x2, y2 = grid_boxes[idx]
    cell_hsv = hsv[y1:y2, x1:x2]
    cell_bgr = bgr[y1:y2, x1:x2]
    h, w = cell_bgr.shape[:2]
    
    print(f"Cell {cell_id} (size: {w}x{h}):")
    
    # Get top-left ROI (40% x 40%)
    tl_roi_hsv = processor._get_roi(cell_hsv, 0, 0, 0.4, 0.4)
    tl_roi_bgr = processor._get_roi(cell_bgr, 0, 0, 0.4, 0.4)
    
    if tl_roi_hsv.size > 0:
        # Analyze all pixels in ROI
        pixels_hsv = tl_roi_hsv.reshape(-1, 3)
        
        # Current mask
        current_mask = cv2.inRange(tl_roi_hsv, 
                                  np.array(processor.COLORS['BLUE']['lower']), 
                                  np.array(processor.COLORS['BLUE']['upper']))
        current_count = cv2.countNonZero(current_mask)
        print(f"  Current blue mask: {current_count} pixels")
        
        # Find all non-black/non-white pixels (likely the badge)
        # White checkmark: high value, low saturation
        # Blue badge: medium-high value, medium-high saturation, blue hue
        gray = cv2.cvtColor(tl_roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Find bright pixels (checkmark or badge)
        bright_mask = gray > 100
        bright_pixels_hsv = pixels_hsv[bright_mask.flatten()]
        
        if len(bright_pixels_hsv) > 0:
            # Filter out white pixels (low saturation)
            # Blue pixels should have saturation > 50 and hue around 100-120
            blue_candidates = bright_pixels_hsv[
                (bright_pixels_hsv[:, 1] > 50) &  # Some saturation
                (bright_pixels_hsv[:, 0] >= 90) & (bright_pixels_hsv[:, 0] <= 130)  # Blue-ish hue
            ]
            
            if len(blue_candidates) > 0:
                h_min, h_max = blue_candidates[:, 0].min(), blue_candidates[:, 0].max()
                s_min, s_max = blue_candidates[:, 1].min(), blue_candidates[:, 1].max()
                v_min, v_max = blue_candidates[:, 2].min(), blue_candidates[:, 2].max()
                
                print(f"  Blue badge pixels found: {len(blue_candidates)}")
                print(f"  Hue range: {h_min:.0f} - {h_max:.0f}")
                print(f"  Sat range: {s_min:.0f} - {s_max:.0f}")
                print(f"  Val range: {v_min:.0f} - {v_max:.0f}")

                # Percentiles help pick a "core blue" that excludes anti-aliased edges
                h_p10, h_p50, h_p90 = np.percentile(blue_candidates[:, 0], [10, 50, 90])
                s_p10, s_p50, s_p90 = np.percentile(blue_candidates[:, 1], [10, 50, 90])
                v_p10, v_p50, v_p90 = np.percentile(blue_candidates[:, 2], [10, 50, 90])
                print(f"  Hue p10/p50/p90: {h_p10:.0f}/{h_p50:.0f}/{h_p90:.0f}")
                print(f"  Sat p10/p50/p90: {s_p10:.0f}/{s_p50:.0f}/{s_p90:.0f}")
                print(f"  Val p10/p50/p90: {v_p10:.0f}/{v_p50:.0f}/{v_p90:.0f}")
                
                # Test with wider range
                test_lower = (max(90, int(h_min - 5)), max(50, int(s_min - 20)), max(100, int(v_min - 30)))
                test_upper = (min(130, int(h_max + 5)), 255, 255)
                test_mask = cv2.inRange(tl_roi_hsv, np.array(test_lower), np.array(test_upper))
                test_count = cv2.countNonZero(test_mask)
                print(f"  Test range {test_lower} - {test_upper}: {test_count} pixels")
        
        # Save ROI for visual inspection
        debug_dir = os.path.join(PROJECT_ROOT, "debug_output")
        os.makedirs(debug_dir, exist_ok=True)
        roi_path = os.path.join(debug_dir, f"badge_analysis_cell{cell_id}_roi.png")
        cv2.imwrite(roi_path, tl_roi_bgr)
        print(f"  Saved ROI to: {roi_path}\n")

