import os
import sys
import pytest
import json

# Add project root to path before imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.image_processor import ImageProcessor
from src.solver import DebugManager

# Paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
RECAPTCHA_DIR = os.path.join(PROJECT_ROOT, "captchaimages", "coreRecaptcha")
ANSWERS_PATH = os.path.join(RECAPTCHA_DIR, "recaptchaAnswers.json")

def test_selected_fields_all_images():
    """
    Test that detected selections match expectations from recaptchaAnswers.json.
    """
    # These checks are now handled in __main__ block, but kept here for pytest compatibility
    if not os.path.exists(RECAPTCHA_DIR):
        pytest.skip(f"Directory not found: {RECAPTCHA_DIR}")
        
    if not os.path.exists(ANSWERS_PATH):
        pytest.skip(f"Answers file not found: {ANSWERS_PATH}")

    with open(ANSWERS_PATH, 'r') as f:
        answers_data = json.load(f)

    # Initialize ImageProcessor with debug enabled
    debug_manager = DebugManager(debug_enabled=False)
    processor = ImageProcessor(attention_extractor=None, planner=None, debug_manager=debug_manager)
    
    failures = []
    
    print(f"\nTesting images in {RECAPTCHA_DIR} against {ANSWERS_PATH}")

    for filename, data in answers_data.items():
        image_path = os.path.join(RECAPTCHA_DIR, filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {filename} not found at {image_path}. Skipping.")
            continue
            
        print(f"Testing {filename}...")
        
        # 1. Detect Grid
        grid_boxes = ImageProcessor.get_grid_bounding_boxes(image_path)
        
        if not grid_boxes:
            # If we expect selections, but can't find a grid, that's a failure.
            # If expected selections are empty, maybe it's fine? 
            # But usually these are grid captchas, so no grid is likely an error in detection or non-grid image.
            # However, looking at the dataset, most should have grids.
            expected_selected = data.get("selectedCells", [])
            if expected_selected:
                 failures.append(f"{filename}: No grid detected, but expected selection {expected_selected}")
            else:
                 print(f"  No grid detected in {filename}. Expecting empty selection. OK.")
            continue
            
        # 2. Detect Selections
        selected_indices, loading_indices = processor.detect_selected_cells(image_path, grid_boxes)
        selected_indices.sort()
        
        expected_selected = data.get("selectedCells", [])
        expected_selected.sort()
        
        # We only verify selectedCells (completed selections) as per request.
        # loading_indices are ignored for this test unless we want to verify them too, 
        # but the JSON only has selectedCells.
        
        if selected_indices != expected_selected:
            error_msg = f"{filename}: Expected {expected_selected}, got {selected_indices}"
            print(f"  FAILED: {error_msg}")
            failures.append(error_msg)
        else:
            print(f"  SUCCESS: Matches {expected_selected}")

    if failures:
        print(f"\n{'='*60}")
        print("TEST FAILURES:")
        print('='*60)
        for failure in failures:
            print(failure)
        print('='*60)
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("ALL TESTS PASSED!")
        print('='*60)

if __name__ == "__main__":
    # When running directly, handle skips differently
    if not os.path.exists(RECAPTCHA_DIR):
        print(f"Error: Directory not found: {RECAPTCHA_DIR}")
        sys.exit(1)
        
    if not os.path.exists(ANSWERS_PATH):
        print(f"Error: Answers file not found: {ANSWERS_PATH}")
        sys.exit(1)
    
    test_selected_fields_all_images()
