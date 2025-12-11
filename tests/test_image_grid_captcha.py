import pytest
import json
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from attention import AttentionExtractor

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

def check_containment(inner, outer, threshold=0.9):
    """
    Check if inner box is contained in outer box.
    Returns True if strictly inside or if intersection/inner area >= threshold.
    """
    # Strict check first
    if (inner[0] >= outer[0] and inner[1] >= outer[1] and 
        inner[2] <= outer[2] and inner[3] <= outer[3]):
        return True

    ix_min = max(inner[0], outer[0])
    iy_min = max(inner[1], outer[1])
    ix_max = min(inner[2], outer[2])
    iy_max = min(inner[3], outer[3])
    
    if ix_max < ix_min or iy_max < iy_min:
        return False # No intersection
        
    intersection_area = (ix_max - ix_min) * (iy_max - iy_min)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    
    if inner_area == 0:
        return False
        
    ratio = intersection_area / inner_area
    return ratio >= threshold

@pytest.fixture(scope="module")
def extractor():
    """Initialize extractor once for all tests."""
    return AttentionExtractor()

# Format: (filename, prompt, valid_region_keys)
GRID_CASES = [
    # Image Grids
    ("hcaptchaImages1.png", "bird", ["topMiddleBirdGroup", "topRightBirdGroup", "bottomLeftBirdGroup"]),
    ("recaptchaImages.png", "car", ["promptCar", "topMiddleCarGroup", "middleRightCarGroup", "bottomRightCar"]),
    ("recaptchaImages2.png", "motorcycle", ["motorcycleContainer"]),
    ("recaptchaImages3.png", "fire hydrant", ["bottomLeftFireHydrant", "middleFireHydrant", "topRightFireHydrant"]),
    
    # Drag Puzzles (Detection based)
    ("hcaptchaDragImage1.png", "top segment movable square", ["topSegment"]),
    ("hcaptchaDragImage2.png", "top movable deer head", ["deerhead"]),
    ("hcaptchaDragImages3.png", "bottom right movable bee", ["bee"]),
    ("hcaptchaDragImages3.png", "top left strawberry", ["beeDesinationStrawberry"]),
]

@pytest.mark.parametrize("filename,prompt,valid_keys", GRID_CASES)
def test_image_detection(extractor, filename, prompt, valid_keys):
    """
    Verify that the AttentionExtractor detects objects within the expected regions.
    Uses 'detect' method.
    """
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        pytest.skip(f"Image not found: {image_path}")
        
    print(f"\nTesting {filename} with prompt '{prompt}'...")
    
    file_config = TARGET_CONFIG.get(filename)
    if not file_config:
        pytest.fail(f"No configuration found for {filename}")
        
    regions_list = file_config.get("target_bounding_boxes") or file_config.get("target_area_percentages")
    
    # Flatten valid boxes map
    valid_boxes = []
    if regions_list:
        for region_map in regions_list:
            for key, box in region_map.items():
                if key in valid_keys:
                    valid_boxes.append((key, box))
    
    if not valid_boxes:
        pytest.fail(f"Could not find valid boxes for keys {valid_keys} in config")
        
    # Execute detect
    result = extractor.detect(image_path, prompt)
    detections = result.get('objects', [])
    print(f"  Found {len(detections)} detections")
    
    # For some difficult cases, we might accept no detections if we want to be lenient, 
    # but the user asked for reliability. So we should assert we found something.
    assert len(detections) > 0, f"No detections found for '{prompt}' in {filename}"
        
    # Check each detection
    # REQUIREMENT: Every prediction/bounding box MUST be inside of our expected bounds
    for det in detections:
        bbox = [det['x_min'], det['y_min'], det['x_max'], det['y_max']]
        
        is_valid = False
        matched_key = None
        for key, valid_box in valid_boxes:
            if check_containment(bbox, valid_box, threshold=0.85): # 85% containment allowed
                is_valid = True
                matched_key = key
                break
        
        if is_valid:
            print(f"  ✓ Detection {bbox} inside {matched_key}")
        else:
            print(f"  ✗ Detection {bbox} outside expected regions")
            
        assert is_valid, f"Detection {bbox} is outside expected regions: {valid_keys}. Found in {filename}"

