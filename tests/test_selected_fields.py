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

def test_selected_fields_recaptcha():
    image_path = os.path.join(IMAGES_DIR, "selectedFieldsRecaptcha.png")
    if not os.path.exists(image_path):
        pytest.skip(f"Image not found: {image_path}")

    # 1. Test Grid Detection
    print(f"\nTesting grid detection on {image_path}...")
    grid_boxes = ImageProcessor.get_grid_bounding_boxes(image_path)
    
    assert grid_boxes is not None, "Failed to detect grid in selectedFieldsRecaptcha.png"
    assert len(grid_boxes) == 9, f"Expected 9 grid cells (3x3), found {len(grid_boxes)}"
    
    # 2. Test Selected Cells Detection
    print(f"Testing selected cells detection...")
    selected_indices = ImageProcessor.detect_selected_cells(image_path, grid_boxes)
    selected_indices.sort()
    
    print(f"Detected selected indices: {selected_indices}")
    expected_indices = [1, 2, 6]
    
    assert selected_indices == expected_indices, f"Expected selected indices {expected_indices}, but got {selected_indices}"

