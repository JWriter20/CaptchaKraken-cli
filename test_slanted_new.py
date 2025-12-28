import os
import sys
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.tool_calls.find_grid import find_grid

def test_slanted_grid():
    image_path = "captchaimages/slantedGrid.png"
    print(f"Testing find_grid on {image_path}...")
    lines = find_grid(image_path)
    
    if lines:
        print(f"Success! Found {len(lines)} line segments.")
        for i, line in enumerate(lines):
            print(f"  Line {i+1}: {line}")
    else:
        print("No lines detected matching the criteria.")
    
    if os.path.exists("red_lines_overlay.png"):
        print("Created red_lines_overlay.png")
    else:
        print("Error: red_lines_overlay.png not created")

if __name__ == "__main__":
    test_slanted_grid()

