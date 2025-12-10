
import sys
import os
import shutil
from unittest.mock import MagicMock
from src.solver import CaptchaSolver
from PIL import Image

def test_solver_foreground_logic():
    print("Initializing Solver (mocked)...")
    solver = CaptchaSolver(provider="ollama")
    solver.planner = MagicMock()
    
    # MOCK: Planner selects ID 1 (Assuming W is 2nd largest area, after background)
    # If this fails to pick the W, we might see the wrong thing, but it proves the logic runs.
    solver.planner.select_items.return_value = [1]
    
    # MOCK: Refine accepts immediately
    solver.planner.refine_drag.return_value = {"decision": "accept", "dx": 0, "dy": 0}
    
    image_path = "captchaimages/sourceWImage.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    with Image.open(image_path) as img:
        solver._image_size = img.size
        w, h = img.size
        print(f"Image size: {w}x{h}")
        
    # Define a source bbox that covers the W part of the image
    # The image has "Move" at top (approx 30px?) and W below.
    # Let's select the bottom part as source bbox
    source_bbox = [0, 30, w, h]
    
    print("Calling _solve_drag...")
    try:
        solver._solve_drag(
            image_path=image_path,
            instruction="Test Drag",
            source_description="white letter W",
            target_description="target",
            source_bbox_override=source_bbox,
            max_iterations=1
        )
        print("Success: _solve_drag finished.")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_solver_foreground_logic()

