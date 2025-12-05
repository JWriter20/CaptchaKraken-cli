#!/usr/bin/env python3
"""
Test script for the new pointing-based coordinate extraction.

Run: python scripts/test_pointing.py
"""

import os
import sys

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from src.captchakraken.attention import AttentionExtractor

# Test images directory
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'captchaimages')

# Test cases: (image_name, target_description, expected_region)
TEST_CASES = [
    ("hcaptchaBasic.png", "the checkbox", "left side"),
    ("recaptchaBasic.png", "the checkbox", "left side"),  
    ("cloudflare.png", "the checkbox or verification button", "center"),
]


def test_single_image(extractor: AttentionExtractor, image_path: str, target: str):
    """Test coordinate extraction on a single image."""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"Target: {target}")
    print('='*60)
    
    try:
        # Extract coordinates (now returns percentages 0-1)
        x_pct, y_pct = extractor.extract_coordinates(image_path, target)
        print(f"\n✓ Position: ({x_pct:.2%}, {y_pct:.2%})")
        
        # Generate visualization
        output_name = f"test_output_{os.path.basename(image_path)}"
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
    print("="*60)
    print("Testing Coordinate Extraction")
    print("="*60)
    
    # Create extractor with moondream backend (uses native point() method)
    print("\nInitializing AttentionExtractor (moondream backend with point())...")
    extractor = AttentionExtractor(
        backend="moondream",
        model="vikhyatk/moondream2"
    )
    
    results = []
    
    for image_name, target, expected in TEST_CASES:
        image_path = os.path.join(IMAGES_DIR, image_name)
        
        if not os.path.exists(image_path):
            print(f"\n⚠ Skipping {image_name} - file not found")
            continue
        
        success, coords = test_single_image(extractor, image_path, target)
        results.append((image_name, success, coords))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for image_name, success, coords in results:
        status = "✓" if success else "✗"
        coord_str = f"({coords[0]:.2%}, {coords[1]:.2%})" if coords else "N/A"
        print(f"{status} {image_name}: {coord_str}")
    
    passed = sum(1 for _, s, _ in results if s)
    print(f"\nPassed: {passed}/{len(results)}")


if __name__ == "__main__":
    main()

