
import pytest
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.attention import AttentionExtractor

# Paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(TESTS_DIR)
IMAGES_DIR = os.path.join(CORE_DIR, "captchaimages")
CONFIG_PATH = os.path.join(IMAGES_DIR, "targetAreaPercentages.json")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, 'r') as f:
        data = json.load(f)
    config = {}
    for item in data:
        config.update(item)
    return config

TARGET_CONFIG = load_config()

def is_point_in_box(point, box):
    """Check if point (x, y) is inside box [x_min, y_min, x_max, y_max]."""
    x, y = point
    # Strict bounds checking
    tol = 0.001
    return (box[0] - tol <= x <= box[2] + tol and 
            box[1] - tol <= y <= box[3] + tol)

def is_box_in_box(inner, outer):
    """Check if inner box is fully inside outer box."""
    # Strict bounds checking
    tol = 0.001
    return (inner[0] >= outer[0] - tol and 
            inner[1] >= outer[1] - tol and 
            inner[2] <= outer[2] + tol and 
            inner[3] <= outer[3] + tol)

# Define the "Ideal" prompts and expected targets
# Format: (filename, method, prompt, valid_region_keys)
# method is "focus" (returns point) or "detect" (returns bboxes)
TEST_CASES = [
    # Checkboxes
    ("cloudflare.png", "focus", "checkbox", ["checkbox"]),
    ("hcaptchaBasic.png", "focus", "checkbox", ["checkbox"]),
    ("recaptchaBasic.png", "focus", "checkbox", ["checkbox"]),
    
    # Image Selection
    ("hcaptchaImages1.png", "detect", "bird", ["topMiddleBirdGroup", "topRightBirdGroup", "bottomLeftBirdGroup"]),
    ("recaptchaImages.png", "detect", "car", ["promptCar", "topMiddleCarGroup", "middleRightCarGroup", "bottomRightCar"]),
    ("recaptchaImages2.png", "detect", "motorcycle", ["motorcycleContainer"]),
    ("recaptchaImages3.png", "detect", "fire hydrant", ["bottomLeftFireHydrant", "middleFireHydrant", "topRightFireHydrant"]),
    
    # Drag Puzzles - Source
    # Note: These prompts correspond to AlgoImprovements.txt logic
    ("hcaptchaDragImage1.png", "detect", "top segment movable square", ["topSegment"]),
    ("hcaptchaDragImage2.png", "detect", "top movable deer head", ["deerhead"]),
    ("hcaptchaDragImages3.png", "detect", "bottom right movable bee", ["bee"]),
    
    # Drag Puzzles - Destination
    ("hcaptchaDragImages3.png", "detect", "top left strawberry", ["beeDesinationStrawberry"]),
]

@pytest.fixture(scope="module")
def extractor():
    """Initialize extractor once for all tests."""
    return AttentionExtractor(backend="moondream")

@pytest.mark.parametrize("filename,method,prompt,valid_keys", TEST_CASES)
def test_real_attention_accuracy(extractor, filename, method, prompt, valid_keys):
    """
    Verify that the AttentionExtractor finds the correct location given an ideal prompt.
    This test runs against REAL models (no mocking).
    """
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        pytest.skip(f"Image not found: {image_path}")
        
    print(f"\nTesting {filename} with prompt '{prompt}' using {method}...")
    
    # Get valid regions from config
    file_config = TARGET_CONFIG.get(filename)
    if not file_config:
        pytest.fail(f"No configuration found for {filename}")
        
    valid_boxes = []
    # Handle both config structures (list of dicts or direct keys)
    # The JSON structure seems to be: "filename": { "target_bounding_boxes": [{"key": [...]}] }
    # OR "target_area_percentages": ...
    
    regions_list = file_config.get("target_bounding_boxes") or file_config.get("target_area_percentages")
    
    # Flatten list of dicts to find matching keys
    for region_map in regions_list:
        for key, box in region_map.items():
            if key in valid_keys:
                valid_boxes.append((key, box))
    
    if not valid_boxes:
        pytest.fail(f"Could not find valid boxes for keys {valid_keys} in config for {filename}")
        
    # Execute extraction
    if method == "focus":
        # Returns (x, y)
        point = extractor.focus(image_path, prompt)
        print(f"  Result point: {point}")
        
        # Check if point is in ANY of the valid boxes
        in_bounds = False
        for key, box in valid_boxes:
            if is_point_in_box(point, box):
                in_bounds = True
                print(f"  ✓ Point inside {key}")
                break
        
        assert in_bounds, f"Focus point {point} not inside any expected regions: {valid_keys}"
        
    elif method == "detect":
        # Returns list of dicts with 'bbox'
        detections = extractor.detect_objects(image_path, prompt)
        print(f"  Found {len(detections)} detections")
        
        if not detections and "Desination" not in valid_keys[0]: 
            # It's okay to miss detections sometimes, but we warn. 
            # STRICT MODE: Fail if nothing found for obvious objects
            pytest.fail(f"No detections found for '{prompt}' in {filename}")
            
        for det in detections:
            bbox = det['bbox']
            # Moondream detect returns [x_min, y_min, x_max, y_max] normalized
            
            # Check if this detection overlaps significantly or is inside a valid box
            # The user requirement: "Every prediction/bounding box MUST be inside of our expected bounds"
            
            is_valid = False
            for key, valid_box in valid_boxes:
                # We use strict containment as requested
                if is_box_in_box(bbox, valid_box):
                    is_valid = True
                    print(f"  ✓ Detection {bbox} inside {key}")
                    break
            
            # If not strictly inside, maybe check intersection? 
            # The user said "without going outside of it". Strict containment.
            # However, model detections might be slightly larger/smaller.
            # We'll stick to strict check but might need to relax if tests represent real model noise.
            # For now, asserting strict containment.
            assert is_valid, f"Detection {bbox} is outside expected regions: {valid_keys}"

