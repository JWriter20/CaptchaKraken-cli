#!/usr/bin/env python3
"""
Test script for the new pointing-based coordinate extraction.

Run: python scripts/test_pointing.py <image_path> <query>
"""

import os
import sys
import argparse

# Add project root and src to path
project_root = os.path.join(os.path.dirname(__file__), '..')
src_root = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_root)

from src.attention import AttentionExtractor

def test_pointing(extractor: AttentionExtractor, image_path: str, target: str, output_path: str = None):
    """Test coordinate extraction on a single image."""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"Target: {target}")
    print('='*60)
    
    try:
        # Extract coordinates (now returns percentages 0-1)
        detections = extractor.detect(image_path, target, max_objects=1)
        if detections:
            obj = detections[0]
            x_pct = (obj["x_min"] + obj["x_max"]) / 2
            y_pct = (obj["y_min"] + obj["y_max"]) / 2
        else:
            x_pct, y_pct = 0.5, 0.5

        print(f"\n✓ Position: ({x_pct:.2%}, {y_pct:.2%})")
        
        # Generate visualization
        if output_path is None:
            # Create test-results directory if it doesn't exist
            results_dir = os.path.join(project_root, "test-results")
            os.makedirs(results_dir, exist_ok=True)
            output_name = os.path.join(results_dir, f"test_pointing_{os.path.basename(image_path)}")
        else:
            output_name = output_path
            
        vis_path = extractor.visualize_attention(
            image_path, 
            target,
            output_path=output_name
        )
        print(f"✓ Visualization saved: {vis_path}")
        
        return True, (x_pct, y_pct)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser(description="Test point extraction on an image using AttentionExtractor.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("query", help="Target description to point to")
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
    
    args = parser.parse_args()

    print("="*60)
    print("Testing Coordinate Extraction")
    print("="*60)
    
    # Create extractor with moondream backend (uses native point() method)
    print(f"\nInitializing AttentionExtractor (model={args.model})...")
    extractor = AttentionExtractor(
        model=args.model,
        device=args.device,
    )
    
    if not os.path.exists(args.image_path):
        print(f"\n✗ Error: Image not found at {args.image_path}")
        return
        
    success, coords = test_pointing(extractor, args.image_path, args.query, args.output)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if success:
        print(f"✓ {os.path.basename(args.image_path)}: ({coords[0]:.2%}, {coords[1]:.2%})")
    else:
        print(f"✗ {os.path.basename(args.image_path)}: Pointing failed")


if __name__ == "__main__":
    main()
