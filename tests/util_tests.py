import pytest
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.attention import AttentionExtractor, clusterDetections

# Paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(TESTS_DIR)
IMAGES_DIR = os.path.join(CORE_DIR, "captchaimages")
CONFIG_PATH = os.path.join(IMAGES_DIR, "targetAreaPercentages.json")

def load_config():
    """Load target area configuration."""
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
    # Strict bounds checking with small tolerance
    tol = 0.001
    return (box[0] - tol <= x <= box[2] + tol and 
            box[1] - tol <= y <= box[3] + tol)

def box_center(box):
    """Calculate center point of a bounding box."""
    if isinstance(box, dict):
        # Dict format: {'x_min': ..., 'y_min': ..., 'x_max': ..., 'y_max': ...}
        return ((box['x_min'] + box['x_max']) / 2, 
                (box['y_min'] + box['y_max']) / 2)
    else:
        # List format: [x_min, y_min, x_max, y_max]
        return ((box[0] + box[2]) / 2, 
                (box[1] + box[3]) / 2)


@pytest.fixture(scope="module")
def extractor():
    """Initialize extractor once for all tests."""
    return AttentionExtractor()


def test_cluster_detections_bird_reduction(extractor):
    """
    Test that clusterDetections() correctly merges overlapping bird detections
    from 11 individual detections down to 3 grouped detections, and that
    the centers of the merged boxes are still within valid regions.
    """
    image_path = os.path.join(IMAGES_DIR, "hcaptchaImages1.png")
    
    if not os.path.exists(image_path):
        pytest.skip(f"Image not found: {image_path}")
    
    print(f"\n=== Testing clusterDetections on hcaptchaImages1.png ===")
    
    # Get valid regions from config
    file_config = TARGET_CONFIG.get("hcaptchaImages1.png")
    if not file_config:
        pytest.fail("No configuration found for hcaptchaImages1.png")
    
    # Extract valid bird regions
    valid_boxes = []
    regions_list = file_config.get("target_bounding_boxes") or file_config.get("target_area_percentages")
    
    valid_keys = ["topMiddleBirdGroup", "topRightBirdGroup", "bottomLeftBirdGroup"]
    
    for region_map in regions_list:
        for key, box in region_map.items():
            if key in valid_keys:
                valid_boxes.append((key, box))
    
    if not valid_boxes:
        pytest.fail(f"Could not find valid boxes for bird groups in config")
    
    print(f"Valid regions: {[key for key, _ in valid_boxes]}")
    
    # Run detection
    print("\n1. Running detection for 'bird'...")
    result = extractor.detect(image_path, "bird", max_objects=12)
    
    detections = result.get('objects', [])
    print(f"   Found {len(detections)} initial detections")
    
    # Verify we got approximately 11 detections (allow some variance)
    assert len(detections) >= 9, f"Expected ~11 detections, got {len(detections)}"
    assert len(detections) <= 13, f"Expected ~11 detections, got {len(detections)}"
    
    print(f"   ✓ Initial detection count: {len(detections)} (expected ~11)")
    
    # Run clustering with distance-based approach for nearby birds
    print("\n2. Running clusterDetections()...")
    # Birds in groups are close but may not overlap much
    # Use distance-based clustering (0.15 = 15% of image width/height)
    clustered = clusterDetections(detections, distance_threshold=0.15)
    
    print(f"   Clustered down to {len(clustered)} detections")
    
    # Verify we got exactly 3 clustered detections (one per bird group)
    assert len(clustered) == 3, f"Expected 3 clustered detections, got {len(clustered)}"
    print(f"   ✓ Clustered detection count: {len(clustered)} (expected 3)")
    
    # Verify centers are in valid regions
    print("\n3. Verifying centers are within valid regions...")
    
    for i, merged_box in enumerate(clustered):
        center = box_center(merged_box)
        print(f"\n   Clustered box {i+1}:")
        print(f"     Box: ({merged_box['x_min']:.3f}, {merged_box['y_min']:.3f}) -> "
              f"({merged_box['x_max']:.3f}, {merged_box['y_max']:.3f})")
        print(f"     Center: ({center[0]:.3f}, {center[1]:.3f})")
        
        # Check if center is in any valid region
        in_valid_region = False
        for key, valid_box in valid_boxes:
            if is_point_in_box(center, valid_box):
                in_valid_region = True
                print(f"     ✓ Center is inside {key}")
                break
        
        assert in_valid_region, (
            f"Merged box center {center} is not in any valid region. "
            f"Valid regions: {valid_keys}"
        )
    
    print("\n=== Test passed! ===")
    print(f"Successfully reduced {len(detections)} detections to {len(clustered)} clustered detections")
    print("All merged box centers are within valid bird group regions")


def test_cluster_detections_car_reduction(extractor):
    """
    Test that clusterDetections() correctly merges overlapping car detections
    from multiple detections down to 2-3 grouped detections, and that
    the centers of the merged boxes are still within valid regions.
    """
    image_path = os.path.join(IMAGES_DIR, "recaptchaImages.png")
    
    if not os.path.exists(image_path):
        pytest.skip(f"Image not found: {image_path}")
    
    print(f"\n=== Testing clusterDetections on recaptchaImages.png ===")
    
    # Get valid regions from config
    file_config = TARGET_CONFIG.get("recaptchaImages.png")
    if not file_config:
        pytest.fail("No configuration found for recaptchaImages.png")
    
    # Extract valid car regions (including the prompt car since model detects it)
    valid_boxes = []
    regions_list = file_config.get("target_bounding_boxes") or file_config.get("target_area_percentages")
    
    # We expect cars in these regions (including promptCar which the model will detect)
    valid_keys = ["promptCar", "topMiddleCarGroup", "middleRightCarGroup", "bottomRightCar"]
    
    for region_map in regions_list:
        for key, box in region_map.items():
            if key in valid_keys:
                valid_boxes.append((key, box))
    
    if not valid_boxes:
        pytest.fail(f"Could not find valid boxes for car groups in config")
    
    print(f"Valid regions: {[key for key, _ in valid_boxes]}")
    
    # Run detection
    print("\n1. Running detection for 'car'...")
    result = extractor.detect(image_path, "car", max_objects=12)
    
    detections = result.get('objects', [])
    print(f"   Found {len(detections)} initial detections")
    
    # Verify we got multiple detections
    assert len(detections) >= 3, f"Expected at least 3 car detections, got {len(detections)}"
    
    print(f"   ✓ Initial detection count: {len(detections)}")
    
    # Run clustering with distance-based approach
    print("\n2. Running clusterDetections()...")
    # Use distance-based clustering for nearby cars
    clustered = clusterDetections(detections, distance_threshold=0.2)
    
    print(f"   Clustered down to {len(clustered)} detections")
    
    # Verify we got 2-3 clustered detections
    assert 2 <= len(clustered) <= 3, f"Expected 2-3 clustered detections, got {len(clustered)}"
    print(f"   ✓ Clustered detection count: {len(clustered)} (expected 2-3)")
    
    # Verify centers are in valid regions
    print("\n3. Verifying centers are within valid regions...")
    
    for i, merged_box in enumerate(clustered):
        center = box_center(merged_box)
        print(f"\n   Clustered box {i+1}:")
        print(f"     Box: ({merged_box['x_min']:.3f}, {merged_box['y_min']:.3f}) -> "
              f"({merged_box['x_max']:.3f}, {merged_box['y_max']:.3f})")
        print(f"     Center: ({center[0]:.3f}, {center[1]:.3f})")
        
        # Check if center is in any valid region
        in_valid_region = False
        for key, valid_box in valid_boxes:
            if is_point_in_box(center, valid_box):
                in_valid_region = True
                print(f"     ✓ Center is inside {key}")
                break
        
        assert in_valid_region, (
            f"Merged box center {center} is not in any valid region. "
            f"Valid regions: {valid_keys}"
        )
    
    print("\n=== Test passed! ===")
    print(f"Successfully reduced {len(detections)} detections to {len(clustered)} clustered detections")
    print("All merged box centers are within valid car group regions")

