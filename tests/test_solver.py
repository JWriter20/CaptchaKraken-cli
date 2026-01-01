import pytest
import os
import sys
import time
import shutil
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.solver import CaptchaSolver
from src.action_types import ClickAction, DragAction, DoneAction
from src.tool_calls.find_grid import find_grid
from src.overlay import add_overlays_to_image

# Ensure debug is on for logging and artifact generation
os.environ["CAPTCHA_DEBUG"] = "1"

# Global solver to avoid wiping debug dir repeatedly
_SOLVER_INSTANCE = None

def get_solver():
    global _SOLVER_INSTANCE
    if _SOLVER_INSTANCE is None:
        _SOLVER_INSTANCE = CaptchaSolver(provider="ollama")
    return _SOLVER_INSTANCE

def setup_module(module):
    """Ensure the debug directory exists and is fresh."""
    debug_dir = Path("latestDebugRun")
    if debug_dir.exists():
        try:
            shutil.rmtree(debug_dir)
        except Exception:
            pass
    debug_dir.mkdir(exist_ok=True, parents=True)

def label_grid_manually(image_path: str, output_name: str):
    """ Helper to manually label a grid and save it to latestDebugRun """
    grid_boxes = find_grid(image_path)
    if not grid_boxes:
        print(f"[Warning] No grid detected for {image_path}")
        return None
    
    overlays = []
    for i, (x1, y1, x2, y2) in enumerate(grid_boxes):
        overlays.append({
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "number": i + 1,
            "color": "#00FF00",
            "box_style": "solid"
        })
    
    output_path = Path("latestDebugRun") / output_name
    add_overlays_to_image(image_path, overlays, output_path=str(output_path), label_position="top-right")
    print(f"[Test] Manually labeled grid saved to {output_path}")
    return grid_boxes

def save_final_result_overlay(image_path, actions, test_name):
    """ Saves an image with the actions overlaid for verification """
    if not actions:
        return
        
    overlays = []
    for i, action in enumerate(actions):
        if isinstance(action, ClickAction):
            bbox = action.target_bounding_box
            # [x1, y1, x2, y2] normalized
            overlays.append({
                "bbox": bbox,
                "number": i + 1,
                "color": "#FF0000",
                "box_style": "dashed"
            })
        elif isinstance(action, DragAction):
            if hasattr(action, 'source_bounding_box') and action.source_bounding_box:
                overlays.append({
                    "bbox": action.source_bounding_box,
                    "text": "source",
                    "color": "#0000FF",
                    "box_style": "solid"
                })
            if hasattr(action, 'target_bounding_box') and action.target_bounding_box:
                overlays.append({
                    "bbox": action.target_bounding_box,
                    "text": "target",
                    "color": "#00FF00",
                    "box_style": "solid"
                })
            
    if overlays:
        output_path = Path("latestDebugRun") / f"final_result_{test_name}.png"
        
        # If image_path is video, solver extracts a frame as 00_base_image.png
        is_video = any(image_path.lower().endswith(ext) for ext in [".mp4", ".webm", ".gif", ".avi"])
        if is_video:
             image_path = "latestDebugRun/00_base_image.png" 
             if not os.path.exists(image_path):
                 print(f"[Warning] Could not find base image for video result: {image_path}")
                 return

        add_overlays_to_image(image_path, overlays, output_path=str(output_path))
        print(f"[Test] Final result overlay saved to {output_path}")

def run_solver_test(image_path, instruction, test_name, expected_action_type=ClickAction, min_actions=0):
    """ Generic solver test runner """
    if not os.path.exists(image_path):
        pytest.skip(f"Image not found: {image_path}")
        
    solver = get_solver()
    
    print(f"\n[Test] Starting solve for {image_path} ({test_name})")
    print(f"[Test] Instruction: {instruction}")
    
    start_time = time.time()
    actions = solver.solve(image_path, instruction)
    end_time = time.time()
    
    print(f"[Test] Inference took {end_time - start_time:.2f} seconds")
    print(f"[Test] Actions returned: {actions}")
    
    # Normalize to list
    if not isinstance(actions, list):
        if isinstance(actions, (ClickAction, DragAction, DoneAction)):
            actions = [actions]
        else:
            actions = []
            
    assert len(actions) >= min_actions, f"Expected at least {min_actions} actions, got {len(actions)}"
    
    # We don't always expect ClickAction (could be DoneAction if no matches)
    # but we check if they are the right types if provided.
    if expected_action_type and actions:
        for action in actions:
            if isinstance(action, DoneAction): continue
            assert isinstance(action, expected_action_type), f"Expected {expected_action_type}, got {type(action)}"
    
    # Save final overlay for user review
    save_final_result_overlay(image_path, actions, test_name)
        
    return actions

def test_3x3_recaptcha():
    """ Test 3x3 reCAPTCHA with manual labeling first """
    image_path = "captchaimages/coreRecaptcha/recaptchaImages.png"
    label_grid_manually(image_path, "manual_label_3x3.png")
    
    instruction = "Select all images with traffic lights"
    run_solver_test(image_path, instruction, "3x3_recaptcha")

def test_4x4_recaptcha():
    """ Test 4x4 reCAPTCHA with manual labeling first """
    image_path = "captchaimages/coreRecaptcha/recaptchaImages2.png"
    label_grid_manually(image_path, "manual_label_4x4.png")
    
    instruction = "Select all squares with traffic lights"
    run_solver_test(image_path, instruction, "4x4_recaptcha")

def test_slanted_grid():
    """ Test slanted reCAPTCHA grid with manual labeling first """
    image_path = "captchaimages/slantedGrid.png"
    label_grid_manually(image_path, "manual_label_slanted.png")
    
    instruction = "Select all squares with crosswalks"
    run_solver_test(image_path, instruction, "slanted_grid")

def test_hcaptcha_puzzle_solve():
    """ Test hcaptchaPuzzle2.png """
    image_path = "captchaimages/hcaptchaPuzzle2.png"
    instruction = "Select the two wire-frame mesh cubes."
    run_solver_test(image_path, instruction, "hcaptcha_puzzle", min_actions=2)

def test_hcaptcha_choose_similar_shapes():
    """ Test hcaptchaChooseSimilarShapes.png """
    image_path = "captchaimages/hcaptchaChooseSimilarShapes.png"
    instruction = "Select the image that contains a similar shape."
    run_solver_test(image_path, instruction, "hcaptcha_similar_shapes")

def test_hcaptcha_drag_image_1():
    """ Test hcaptchaDragImage1.png (Drag puzzle) """
    image_path = "captchaimages/hcaptchaDragImage1.png"
    instruction = "Drag the puzzle piece to the center of the matching shadow."
    run_solver_test(image_path, instruction, "hcaptcha_drag_1", expected_action_type=DragAction)

def test_hcaptcha_drag_images_3():
    """ Test hcaptchaDragImages3.png (Drag puzzle) """
    image_path = "captchaimages/hcaptchaDragImages3.png"
    instruction = "Complete the puzzle by dragging the piece."
    run_solver_test(image_path, instruction, "hcaptcha_drag_3", expected_action_type=DragAction)

def test_hcaptcha_video_webm():
    """ Test video solving with hcaptcha_1766539373078.webm """
    video_path = "captchaimages/hcaptcha_1766539373078.webm"
    instruction = "Select the specify object in the video."
    run_solver_test(video_path, instruction, "hcaptcha_video")

if __name__ == "__main__":
    # Ensure setup is called if running directly
    setup_module(None)
    pytest.main([__file__, "-s"])
