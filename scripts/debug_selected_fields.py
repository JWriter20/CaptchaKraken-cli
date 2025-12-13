#!/usr/bin/env python3
"""Debug script for selected field detection"""
import os
import sys
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.image_processor import ImageProcessor
from src.solver import DebugManager

# Test image
image_path = os.path.join(PROJECT_ROOT, "captchaimages", "coreRecaptcha", "motorcycle3of7Selected.png")

print(f"Testing: {image_path}")
print(f"Image exists: {os.path.exists(image_path)}")

# Initialize with debug enabled
debug_manager = DebugManager(debug_enabled=True)
processor = ImageProcessor(attention_extractor=None, planner=None, debug_manager=debug_manager)

# 1. Detect grid
print("\n=== Detecting Grid ===")
grid_boxes = ImageProcessor.get_grid_bounding_boxes(image_path)
print(f"Found {len(grid_boxes)} grid cells")

if not grid_boxes:
    print("ERROR: No grid detected!")
    sys.exit(1)

# 2. Detect selected cells
print("\n=== Detecting Selected Cells ===")
selected_indices, loading_indices = processor.detect_selected_cells(image_path, grid_boxes)
print(f"Selected: {selected_indices}")
print(f"Loading: {loading_indices}")

# 3. Manual inspection of cells 6, 10, 11
print("\n=== Manual Inspection of Expected Cells ===")
expected = [6, 10, 11]
img = cv2.imread(image_path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

for cell_id in expected:
    idx = cell_id - 1  # Convert to 0-based
    if idx < 0 or idx >= len(grid_boxes):
        print(f"Cell {cell_id}: Index out of range")
        continue
    
    x1, y1, x2, y2 = grid_boxes[idx]
    cell_hsv = hsv[y1:y2, x1:x2]
    cell_bgr = img[y1:y2, x1:x2]
    h, w = cell_bgr.shape[:2]
    
    print(f"\nCell {cell_id} (box: {x1},{y1} to {x2},{y2}, size: {w}x{h}):")
    
    # Check top-left ROI
    tl_roi = processor._get_roi(cell_hsv, 0, 0, 0.4, 0.4)
    tl_roi_bgr = processor._get_roi(cell_bgr, 0, 0, 0.4, 0.4)
    
    if tl_roi.size > 0:
        # Check for blue pixels
        blue_mask = cv2.inRange(tl_roi, np.array(processor.COLORS['BLUE']['lower']), np.array(processor.COLORS['BLUE']['upper']))
        pixel_count = cv2.countNonZero(blue_mask)
        threshold = tl_roi.size * 0.003
        print(f"  Top-left ROI: {tl_roi.shape[1]}x{tl_roi.shape[0]}, size={tl_roi.size}")
        print(f"  Blue pixels: {pixel_count}, threshold: {threshold:.1f}")
        
        if pixel_count > threshold:
            # Check circularity
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"  Found {len(contours)} contours")
            if contours:
                for i, cnt in enumerate(contours):
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    print(f"    Contour {i}: area={area}, perimeter={perimeter}, points={len(cnt)}")
                
                max_cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(max_cnt)
                perimeter = cv2.arcLength(max_cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    print(f"  Max contour - Circularity: {circularity:.3f} (required: >0.85)")
                    print(f"  Max contour - Area: {area}, Perimeter: {perimeter}")
                    
                    # Try morphological operations to connect pixels
                    kernel = np.ones((3, 3), np.uint8)
                    dilated_mask = cv2.dilate(blue_mask, kernel, iterations=1)
                    dilated_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if dilated_contours:
                        max_dilated = max(dilated_contours, key=cv2.contourArea)
                        dilated_area = cv2.contourArea(max_dilated)
                        dilated_perimeter = cv2.arcLength(max_dilated, True)
                        if dilated_perimeter > 0:
                            dilated_circularity = 4 * np.pi * dilated_area / (dilated_perimeter * dilated_perimeter)
                            print(f"  After dilation - Circularity: {dilated_circularity:.3f}, Area: {dilated_area}, Perimeter: {dilated_perimeter}")
                else:
                    print(f"  Zero perimeter!")
            else:
                print(f"  No contours found")
        else:
            print(f"  Pixel count too low: {pixel_count} <= {threshold:.1f}")
        
        # Save ROI for inspection
        debug_dir = os.path.join(PROJECT_ROOT, "debug_output")
        os.makedirs(debug_dir, exist_ok=True)
        roi_path = os.path.join(debug_dir, f"motorcycle3of7Selected.png_cell{cell_id}_roi.png")
        cv2.imwrite(roi_path, tl_roi_bgr)
        mask_path = os.path.join(debug_dir, f"motorcycle3of7Selected.png_cell{cell_id}_mask.png")
        cv2.imwrite(mask_path, blue_mask)
        print(f"  Saved ROI to: {roi_path}")
        print(f"  Saved mask to: {mask_path}")

