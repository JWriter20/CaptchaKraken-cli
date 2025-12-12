
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.image_processor import ImageProcessor

def test_cv_improvements():
    processor = ImageProcessor()
    
    # Test 1: Fire Hydrant (False Positive Check)
    # Expectation: NO selected cells (User reported false positives on 6 and 8)
    img_hydrant = os.path.join("captchaimages", "fireHydrantCaptcha.png")
    if os.path.exists(img_hydrant):
        print(f"\n--- Testing {os.path.basename(img_hydrant)} ---")
        grid_boxes = processor.get_grid_bounding_boxes(img_hydrant)
        if grid_boxes:
            selected, loading = processor.detect_selected_cells(img_hydrant, grid_boxes)
            print(f"Selected: {selected}")
            print(f"Loading: {loading}")
            if not selected:
                print("PASS: No false positives on Fire Hydrant.")
            else:
                print(f"FAIL: False positives detected on Fire Hydrant: {selected}")
        else:
            print("FAIL: Could not detect grid on Fire Hydrant.")
    else:
        print(f"SKIP: {img_hydrant} not found.")

    # Test 2: Selected Fields (True Positive Check)
    # Expectation: Cells 1, 2, 6 should be selected (indices 0, 1, 5) -> Cells 1, 2, 6
    # Looking at the provided image in chat, the checkmarks are on:
    # Row 1 Col 1 (1)
    # Row 1 Col 2 (2)
    # Row 2 Col 3 (6)
    img_selected = os.path.join("captchaimages", "selectedFieldsRecaptcha.png")
    if os.path.exists(img_selected):
        print(f"\n--- Testing {os.path.basename(img_selected)} ---")
        grid_boxes = processor.get_grid_bounding_boxes(img_selected)
        if grid_boxes:
            selected, loading = processor.detect_selected_cells(img_selected, grid_boxes)
            print(f"Selected: {selected}")
            
            expected = [1, 2, 6] 
            # Note: Depending on detection, it might miss some if I tighten too much.
            missing = [x for x in expected if x not in selected]
            extra = [x for x in selected if x not in expected]
            
            if not missing and not extra:
                print("PASS: Correctly identified all selections.")
            elif not missing:
                print(f"WARN: Identified all expected but found extras: {extra}")
            else:
                print(f"FAIL: Missing expected selections: {missing}")
        else:
            print("FAIL: Could not detect grid on Selected Fields.")
    else:
        print(f"SKIP: {img_selected} not found.")

if __name__ == "__main__":
    test_cv_improvements()

