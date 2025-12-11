import os
import sys
import pytest
from src.image_processor import ImageProcessor

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(TESTS_DIR)
IMAGES_DIR = os.path.join(CORE_DIR, "captchaimages")

def test_selected_fields_all_images():
    """
    Test that selectedFieldsRecaptcha.png is the ONLY image with detected selections.
    All other images should return empty lists.
    """
    if not os.path.exists(IMAGES_DIR):
        pytest.skip("captchaimages directory not found")

    images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Define expectations
    # Filename -> List of 1-based indices
    expectations = {
        "selectedFieldsRecaptcha.png": [1, 2, 6]
    }

    for filename in images:
        image_path = os.path.join(IMAGES_DIR, filename)
        print(f"\nTesting {filename}...")
        
        # 1. Detect Grid
        grid_boxes = ImageProcessor.get_grid_bounding_boxes(image_path)
        
        if not grid_boxes:
            print(f"  No grid detected in {filename}. Skipping selection check.")
            continue
            
        # 2. Detect Selections
        # Initialize ImageProcessor (mock dependencies as they are not needed for this method)
        processor = ImageProcessor(attention_extractor=None, planner=None, debug_manager=None)
        selected_indices = processor.detect_selected_cells(image_path, grid_boxes)
        selected_indices.sort()
        
        print(f"  Detected: {selected_indices}")
        
        expected = expectations.get(filename, [])
        expected.sort()
        
        # Determine if this failure is critical
        # For selectedFieldsRecaptcha.png, we MUST match [1, 2, 6]
        # For others, we MUST match []
        
        if filename == "selectedFieldsRecaptcha.png":
            assert selected_indices == expected, f"Failed for {filename}: Expected {expected}, got {selected_indices}"
        else:
            # For other images, we verify no false positives
            assert selected_indices == [], f"False positive in {filename}: Found {selected_indices}, expected empty"
