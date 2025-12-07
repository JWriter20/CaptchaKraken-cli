
import pytest
import os
import sys
import json
import shutil
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.planner import ActionPlanner
from src.overlay import add_drag_overlay

# Paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(TESTS_DIR)
IMAGES_DIR = os.path.join(CORE_DIR, "captchaimages")

# Define Test Cases
# (filename, context_prompt, expected_kind, expected_partial_response)
TEST_CASES = [
    # Checkbox
    ("cloudflare.png", "Solve this captcha", "checkbox", {}),
    ("hcaptchaBasic.png", "Solve this captcha", "checkbox", {}),
    
    # Image Selection
    ("recaptchaImages.png", "Select all images with cars", "images", 
     {"target_to_detect": "car"}),
     
    ("hcaptchaImages1.png", "Select all images with birds", "images", 
     {"target_to_detect": "bird"}),
     
    # Drag Puzzles (Logical)
    ("hcaptchaDragImages3.png", "Drag the bee to the strawberry", "drag_puzzle", 
     {"drag_type": "logical", "draggable_prompt": "bee", "destination_prompt": "strawberry"}),
     
    ("hcaptchaDragImage2.png", "Drag the deer", "drag_puzzle", 
     {"drag_type": "template_matching", "draggable_prompt": "deer head"}),
]

@pytest.fixture(scope="module")
def planner():
    return ActionPlanner(backend="ollama", thinking_enabled=False) # Disable thinking for faster/cleaner JSON parsing tests

@pytest.mark.parametrize("filename,context,expected_kind,expected_params", TEST_CASES)
def test_real_planner_reasoning(planner, filename, context, expected_kind, expected_params):
    """
    Verify that the ActionPlanner correctly classifies and plans the captcha solution.
    This runs against REAL models (Ollama).
    """
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        pytest.skip(f"Image not found: {image_path}")
        
    print(f"\nTesting {filename} with context '{context}'...")
    
    # 1. Test Classification
    classification = planner.classify_captcha(image_path, context)
    print(f"  Classification: {classification}")
    
    actual_kind = classification.get("captcha_kind")
    assert actual_kind == expected_kind, f"Expected kind {expected_kind}, got {actual_kind}"
    
    # 2. Test Specialized Planning based on kind
    if expected_kind == "images":
        target_plan = planner.plan_detection_target(image_path, context, captcha_kind=expected_kind)
        print(f"  Target Plan: {target_plan}")
        
        target = target_plan.get("target_to_detect", "").lower()
        expected_target = expected_params.get("target_to_detect", "").lower()
        
        assert expected_target in target, f"Expected target '{expected_target}' to be in '{target}'"
        
    elif expected_kind == "drag_puzzle":
        drag_plan = planner.plan_drag_strategy(image_path, context)
        print(f"  Drag Plan: {drag_plan}")
        
        # Check Drag Type
        if "drag_type" in expected_params:
            expected_types = expected_params["drag_type"]
            if isinstance(expected_types, str):
                expected_types = [expected_types]
            assert drag_plan.get("drag_type") in expected_types
            
        # Check Prompts contain keywords
        if "draggable_prompt" in expected_params:
            actual = drag_plan.get("draggable_prompt", "").lower()
            expected = expected_params["draggable_prompt"].lower()
            assert expected in actual, f"Expected draggable prompt to contain '{expected}', got '{actual}'"
            
        if "destination_prompt" in expected_params:
            actual = drag_plan.get("destination_prompt", "").lower()
            expected = expected_params["destination_prompt"].lower()
            assert expected in actual, f"Expected destination prompt to contain '{expected}', got '{actual}'"


def test_drag_destination_diagnostics_deer_head(planner, tmp_path):
    """
    Diagnostics for the deer-head drag puzzle to catch fallback 0.5,0.5 outputs.
    We highlight the head, ask the planner for move coordinates, and save an
    overlay showing the proposed move.
    """
    image_path = os.path.join(IMAGES_DIR, "hcaptchaDragImage2.png")
    if not os.path.exists(image_path):
        pytest.skip(f"Image not found: {image_path}")

    # Ground-truth source box (normalized) from targetAreaPercentages.json
    source_bbox_pct = [0.2375, 0.2053, 0.4131, 0.378]

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    def pct_to_px(bbox_pct):
        x1, y1, x2, y2 = bbox_pct
        return [x1 * img_width, y1 * img_height, x2 * img_width, y2 * img_height]

    source_bbox_px = pct_to_px(source_bbox_pct)
    source_center_pct = (
        (source_bbox_pct[0] + source_bbox_pct[2]) / 2,
        (source_bbox_pct[1] + source_bbox_pct[3]) / 2,
    )

    work_path = tmp_path / "deer_head_overlay.png"
    shutil.copyfile(image_path, work_path)
    add_drag_overlay(str(work_path), source_bbox_px)
    
    initial_plan = planner.get_drag_destination(str(work_path), prompt_text="Move the draggable piece to the correct spot")
    print(f"Initial drag plan: {json.dumps(initial_plan, default=str)}")
    
    target_pct = [
        initial_plan.get("target_x"),
        initial_plan.get("target_y"),
    ]
    
    assert all(v is not None for v in target_pct), "Model did not return target coordinates"

    # Clamp to image bounds and guard against the fallback center
    target_pct = [
        max(0.0, min(1.0, target_pct[0])),
        max(0.0, min(1.0, target_pct[1])),
    ]
    assert not (
        abs(target_pct[0] - 0.5) < 1e-3 and abs(target_pct[1] - 0.5) < 1e-3
    ), "Model returned fallback center (0.5, 0.5) instead of a directed move"

    w_pct = source_bbox_pct[2] - source_bbox_pct[0]
    h_pct = source_bbox_pct[3] - source_bbox_pct[1]
    target_bbox_pct = [
        max(0.0, target_pct[0] - w_pct / 2),
        max(0.0, target_pct[1] - h_pct / 2),
        min(1.0, target_pct[0] + w_pct / 2),
        min(1.0, target_pct[1] + h_pct / 2),
    ]
    target_bbox_px = pct_to_px(target_bbox_pct)
    target_center_px = (target_pct[0] * img_width, target_pct[1] * img_height)

    add_drag_overlay(
        str(work_path),
        source_bbox_px,
        target_bbox=target_bbox_px,
        target_center=target_center_px,
    )

    print(f"Diagnostic overlay saved to {work_path}")

