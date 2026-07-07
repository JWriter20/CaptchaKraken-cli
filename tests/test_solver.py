import pytest
import os
import sys
import time
import shutil
import json
import multiprocessing
from pathlib import Path

# Set start method to 'spawn' for CUDA multiprocessing compatibility
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

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
_TARGET_DATA = None

def get_target_data():
    global _TARGET_DATA
    if _TARGET_DATA is None:
        json_path = Path("captchaimages/targetAreaPercentages.json")
        if json_path.exists():
            with open(json_path, "r") as f:
                raw_data = json.load(f)
                # Flatten the list of dicts if it is a list
                if isinstance(raw_data, list):
                    _TARGET_DATA = {}
                    for item in raw_data:
                        _TARGET_DATA.update(item)
                else:
                    _TARGET_DATA = raw_data
        else:
            _TARGET_DATA = {}
    return _TARGET_DATA

def get_solver():
    global _SOLVER_INSTANCE
    if _SOLVER_INSTANCE is None:
        _SOLVER_INSTANCE = CaptchaSolver(
            provider="captchaKrakenApi",
            model="Qwen/Qwen3-VL-8B-Instruct"
        )
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

def verify_actions(image_path, actions):
    """ Verifies actions against targetAreaPercentages.json """
    target_data = get_target_data()
    filename = os.path.basename(image_path)
    
    if filename not in target_data:
        print(f"[Test] No ground truth data for {filename}, skipping verification.")
        return True

    ground_truth = target_data[filename]
    # Handle both keys
    targets = ground_truth.get("target_bounding_boxes") or ground_truth.get("target_area_percentages")
    if not targets:
        print(f"[Test] No targets found in ground truth for {filename}.")
        return True

    # targets is a list of dicts, let's flatten it to a list of bounding boxes
    gt_boxes = {}
    for t_dict in targets:
        for name, bbox in t_dict.items():
            gt_boxes[name] = bbox

    print(f"[Test] Verifying against {len(gt_boxes)} ground truth targets for {filename}...")
    
    success = True
    matched_gt_names = set()

    for action in actions:
        if isinstance(action, ClickAction):
            for i, pred_bbox in enumerate(action.target_bounding_boxes):
                # pred_bbox is [x1, y1, x2, y2] normalized
                # Check if it overlaps with ANY gt_box
                matched = False
                for name, gt_bbox in gt_boxes.items():
                    # gt_bbox is [x1, y1, x2, y2] normalized
                    if (pred_bbox[0] < gt_bbox[2] and pred_bbox[2] > gt_bbox[0] and
                        pred_bbox[1] < gt_bbox[3] and pred_bbox[3] > gt_bbox[1]):
                        print(f"  [Match] Predicted click {i+1} hits target '{name}'")
                        matched = True
                        matched_gt_names.add(name)
                        break
                if not matched:
                    print(f"  [Failure] Predicted click {i+1} {pred_bbox} did not hit any ground truth targets.")
                    success = False
        elif isinstance(action, DragAction):
            # Verify source
            source_matched = False
            if action.source_bounding_box:
                for name, gt_bbox in gt_boxes.items():
                    if (action.source_bounding_box[0] < gt_bbox[2] and action.source_bounding_box[2] > gt_bbox[0] and
                        action.source_bounding_box[1] < gt_bbox[3] and action.source_bounding_box[3] > gt_bbox[1]):
                        print(f"  [Match] Drag source hits target '{name}'")
                        source_matched = True
                        matched_gt_names.add(name)
                        break
            if not source_matched:
                print(f"  [Failure] Drag source {action.source_bounding_box} did not hit any ground truth targets.")
                success = False

            # Verify target
            target_matched = False
            if action.target_bounding_box:
                for name, gt_bbox in gt_boxes.items():
                    if (action.target_bounding_box[0] < gt_bbox[2] and action.target_bounding_box[2] > gt_bbox[0] and
                        action.target_bounding_box[1] < gt_bbox[3] and action.target_bounding_box[3] > gt_bbox[1]):
                        print(f"  [Match] Drag target hits target '{name}'")
                        target_matched = True
                        matched_gt_names.add(name)
                        break
            if not target_matched:
                print(f"  [Failure] Drag target {action.target_bounding_box} did not hit any ground truth targets.")
                success = False
        elif isinstance(action, DoneAction):
            # DoneAction is always valid if returned by solver
            pass
            
    # Check for missed ground truth targets that are NOT prompts or containers
    missed_gt = set(gt_boxes.keys()) - matched_gt_names
    real_targets_missed = [n for n in missed_gt if not any(x in n.lower() for x in ["prompt", "container", "desination", "target"])]
    
    if real_targets_missed:
        print(f"  [Warning] Missed these ground truth targets: {real_targets_missed}")
        # For now, we don't fail the test if we miss a target, as long as everything we clicked WAS a target.
        # This is because the solver might decide some things are not matches even if we labeled them.
        # But for reCAPTCHA/hCAPTCHA grids, we might want to be stricter.
    
    return success

def print_debug_report():
    """ Prints the log from latestDebugRun/log.txt """
    log_path = Path("latestDebugRun/log.txt")
    if log_path.exists():
        print("\n" + "="*50)
        print("DETAILED MODEL REPORT")
        print("="*50)
        with open(log_path, "r") as f:
            print(f.read())
        print("="*50 + "\n")

def save_final_result_overlay(image_path, actions, test_name):
    """ Saves an image with the actions overlaid for verification """
    if not actions:
        return
        
    overlays = []
    for i, action in enumerate(actions):
        if isinstance(action, ClickAction):
            for j, bbox in enumerate(action.target_bounding_boxes):
                # [x1, y1, x2, y2] normalized
                overlays.append({
                    "bbox": bbox,
                    "number": j + 1,
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

def run_solver_test(image_path, test_name, expected_action_type=ClickAction, min_actions=0):
    """ Generic solver test runner """
    if not os.path.exists(image_path):
        pytest.skip(f"Image not found: {image_path}")
        
    solver = get_solver()
    
    print(f"\n[Test] Starting solve for {image_path} ({test_name})")
    
    start_time = time.time()
    actions = solver.solve(image_path)
    end_time = time.time()
    
    print(f"[Test] Inference took {end_time - start_time:.2f} seconds")
    
    # Normalize to list
    if not isinstance(actions, list):
        if isinstance(actions, (ClickAction, DragAction, DoneAction)):
            actions = [actions]
        else:
            actions = []
            
    print(f"[Test] Actions returned: {actions}")

    # Print debug report for each test
    print_debug_report()

    # Verify actions
    verify_success = verify_actions(image_path, actions)
    
    # Calculate total elements to compare with min_actions
    total_elements = 0
    for action in actions:
        if isinstance(action, ClickAction):
            total_elements += len(action.target_bounding_boxes)
        elif isinstance(action, DragAction):
            total_elements += 1
        # DoneAction and WaitAction don't count as "actions" for min_actions

    assert total_elements >= min_actions, f"Expected at least {min_actions} elements, got {total_elements}"
    
    # We don't always expect ClickAction (could be DoneAction if no matches)
    # but we check if they are the right types if provided.
    if expected_action_type and actions:
        for action in actions:
            if isinstance(action, DoneAction): continue
            assert isinstance(action, expected_action_type), f"Expected {expected_action_type}, got {type(action)}"
    
    # Save final overlay for user review
    save_final_result_overlay(image_path, actions, test_name)
    
    assert verify_success, "Verification against ground truth failed."
        
    return actions

def test_3x3_recaptcha():
    """ Test 3x3 reCAPTCHA with manual labeling first """
    image_path = "captchaimages/coreRecaptcha/recaptchaImages.png"
    label_grid_manually(image_path, "manual_label_3x3.png")
    
    run_solver_test(image_path, "3x3_recaptcha")

def test_4x4_recaptcha():
    """ Test 4x4 reCAPTCHA with manual labeling first """
    image_path = "captchaimages/coreRecaptcha/recaptchaImages2.png"
    label_grid_manually(image_path, "manual_label_4x4.png")
    run_solver_test(image_path, "4x4_recaptcha")

def test_slanted_grid():
    """ Test slanted reCAPTCHA grid with manual labeling first """
    image_path = "captchaimages/slantedGrid.png"
    label_grid_manually(image_path, "manual_label_slanted.png")
    run_solver_test(image_path, "slanted_grid")

def test_hcaptcha_puzzle_solve():
    """ Test hcaptchaPuzzle2.png """
    image_path = "captchaimages/hcaptchaPuzzle2.png"
    run_solver_test(image_path, "hcaptcha_puzzle", min_actions=2)

def test_hcaptcha_choose_similar_shapes():
    """ Test hcaptchaChooseSimilarShapes.png """
    image_path = "captchaimages/hcaptchaChooseSimilarShapes.png"
    run_solver_test(image_path, "hcaptcha_similar_shapes")

def test_hcaptcha_drag_image_1():
    """ Test hcaptchaDragImage1.png (Drag puzzle) """
    image_path = "captchaimages/hcaptchaDragImage1.png"
    run_solver_test(image_path, "hcaptcha_drag_1", expected_action_type=DragAction)

def test_hcaptcha_drag_images_3():
    """ Test hcaptchaDragImages3.png (Drag puzzle) """
    image_path = "captchaimages/hcaptchaDragImages3.png"
    run_solver_test(image_path, "hcaptcha_drag_3", expected_action_type=DragAction)

def test_hcaptcha_video_webm():
    """ Test video solving with hcaptcha_1766539373078.webm """
    video_path = "captchaimages/hcaptcha_1766539373078.webm"
    run_solver_test(video_path, "hcaptcha_video")

def test_cloudflare_checkbox():
    """ Test cloudflare.png """
    image_path = "captchaimages/cloudflare.png"
    run_solver_test(image_path, "cloudflare_checkbox")

def test_hcaptcha_basic():
    """ Test hcaptchaBasic.png """
    image_path = "captchaimages/hcaptchaBasic.png"
    run_solver_test(image_path, "hcaptcha_basic")

def test_recaptcha_basic():
    """ Test recaptchaBasic.png """
    image_path = "captchaimages/recaptchaBasic.png"
    run_solver_test(image_path, "recaptcha_basic")

def test_hcaptcha_images_1():
    """ Test hcaptchaImages1.png """
    image_path = "captchaimages/hcaptchaImages1.png"
    run_solver_test(image_path, "hcaptcha_images_1")

def test_recaptcha_images_3():
    """ Test recaptchaImages3.png """
    image_path = "captchaimages/coreRecaptcha/recaptchaImages3.png"
    run_solver_test(image_path, "recaptcha_images_3")

def test_hcaptcha_drag_2():
    """ Test hcaptchaDragImage2.png """
    image_path = "captchaimages/hcaptchaDragImage2.png"
    run_solver_test(image_path, "hcaptcha_drag_2", expected_action_type=DragAction)

if __name__ == "__main__":
    # Ensure setup is called if running directly
    # test_hcaptcha_drag_images_3()
    setup_module(None)
    pytest.main([__file__, "-s"])
