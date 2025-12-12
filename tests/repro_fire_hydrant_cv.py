
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.image_processor import ImageProcessor

def test_fire_hydrant_false_positives():
    processor = ImageProcessor()
    
    # Path to the image causing issues
    image_path = os.path.join("captchaimages", "fireHydrantCaptcha.png")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Testing detection on {image_path}")
    
    # Get grid boxes (using existing method or mocking if needed, but the processor has detection)
    # We need to call detect_selected_cells. It requires grid_boxes.
    # We can use get_grid_bounding_boxes first.
    
    grid_boxes = processor.get_grid_bounding_boxes(image_path)
    
    if not grid_boxes:
        print("Failed to detect grid boxes.")
        return
        
    print(f"Detected {len(grid_boxes)} grid cells.")
    
    # Run detection
    selected, loading = processor.detect_selected_cells(image_path, grid_boxes)
    
    print(f"Selected cells: {selected}")
    print(f"Loading cells: {loading}")
    
    # Based on user report, we expect 6 and 8 to be FALSELY detected as selected in the current version.
    # If the fix works, selected should be empty [].
    
    if 6 in selected:
        print("FAILURE: Cell 6 falsely detected as selected!")
    if 8 in selected:
        print("FAILURE: Cell 8 falsely detected as selected!")
        
    if not selected:
        print("SUCCESS: No false positives detected.")

if __name__ == "__main__":
    test_fire_hydrant_false_positives()

