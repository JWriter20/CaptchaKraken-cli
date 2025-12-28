#!/usr/bin/env python3
"""
Test script for moondream's detect() method.

Run: python scripts/test_detection.py <image_path> <query>
"""

import os
import sys
import argparse
import importlib.util

# Add project root and src to path
project_root = os.path.join(os.path.dirname(__file__), '..')
src_root = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_root)

from src.attention import AttentionExtractor


def test_detection(extractor: AttentionExtractor, image_path: str, object_class: str, output_path: str = None, max_objects: int = 24):
    """Test object detection on a single image."""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"Object Class: {object_class}")
    print(f"Max Objects: {max_objects}")
    print('='*60)
    
    try:
        # Detect objects using detect() method
        detections = extractor.detect(image_path, object_class, max_objects=max_objects)
        print(f"\n✓ Found {len(detections)} detection(s)")
        
        # Print detection details
        formatted_detections = []
        for i, det in enumerate(detections):
            bbox = [det['x_min'], det['y_min'], det['x_max'], det['y_max']]
            label = object_class
            print(f"  Detection {i+1}:")
            print(f"    Label: {label}")
            print(f"    BBox: [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
            print(f"    Center: ({(bbox[0] + bbox[2])/2:.4f}, {(bbox[1] + bbox[3])/2:.4f})")
            formatted_detections.append({
                "bbox": bbox,
                "score": det["score"]
            })
        
        # Generate visualization
        if output_path is None:
            # Create test-results directory if it doesn't exist
            results_dir = os.path.join(project_root, "test-results")
            os.makedirs(results_dir, exist_ok=True)
            output_name = os.path.join(results_dir, f"test_detection_{os.path.basename(image_path)}")
        else:
            output_name = output_path
            
        vis_path = extractor.visualize_detections(
            image_path,
            formatted_detections,
            output_path=output_name
        )
        print(f"✓ Visualization saved: {vis_path}")
        
        return True, formatted_detections
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser(description="Test object detection on an image using AttentionExtractor.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("query", help="Object class to detect")
    parser.add_argument("--output", "-o", help="Output path for visualization", default=None)
    parser.add_argument(
        "--model",
        "-m",
        help="Hugging Face model id (default: vikhyatk/moondream2)",
        default="vikhyatk/moondream2",
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Device to run on (cuda, mps, cpu). Leave empty to auto-detect.",
        default=None,
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=24,
        help="Maximum number of objects to detect (default: 24)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Testing moondream detect() Method")
    print("="*60)

    # Create extractor with moondream
    print(f"\nInitializing AttentionExtractor (model={args.model})...")
    extractor = AttentionExtractor(
        model=args.model,
        device=args.device,
    )
    
    if not os.path.exists(args.image_path):
        print(f"\n✗ Error: Image not found at {args.image_path}")
        return
    
    # We need to pass max_objects to test_detection, but test_detection needs to be updated too
    # Actually, test_detection calls extractor.detect_objects, so we can just modify test_detection
    # to accept max_objects
    success, detections = test_detection(extractor, args.image_path, args.query, args.output, args.max_objects)
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if success:
        print(f"✓ {os.path.basename(args.image_path)}: Found {len(detections)} detection(s)")
        for i, det in enumerate(detections):
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            print(f"  Detection {i+1}: Center at ({center_x:.2%}, {center_y:.2%})")
    else:
        print(f"✗ {os.path.basename(args.image_path)}: Moondream detection failed")


if __name__ == "__main__":
    main()
