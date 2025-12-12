
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.image_processor import ImageProcessor

def test_loading_detection():
    image_path = "captchaimages/selectedDisappearingRecaptcha.png"
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    processor = ImageProcessor()
    
    # 1. Get grid boxes
    print(f"Detecting grid in {image_path}...")
    grid_boxes = processor.get_grid_bounding_boxes(image_path)
    
    if not grid_boxes:
        print("Error: No grid detected.")
        return
        
    print(f"Detected {len(grid_boxes)} grid cells.")
    
    # 2. Detect selected/loading cells
    print("Running detect_selected_cells...")
    selected_indices, loading_indices = processor.detect_selected_cells(image_path, grid_boxes)
    
    print("-" * 30)
    print(f"Selected Indices (Top-Left): {selected_indices}")
    print(f"Loading Indices (Center):    {loading_indices}")
    print("-" * 30)
    
    # 3. Validation
    target_cell = 5
    if target_cell in loading_indices:
        print(f"SUCCESS: Cell {target_cell} correctly identified as LOADING.")
    else:
        print(f"FAILURE: Cell {target_cell} NOT identified as loading.")
        
    if target_cell in selected_indices:
        print(f"WARNING: Cell {target_cell} was ALSO identified as SELECTED (Top-Left).")

if __name__ == "__main__":
    test_loading_detection()

