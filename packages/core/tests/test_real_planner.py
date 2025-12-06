
import pytest
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from captchakraken.planner import ActionPlanner

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

