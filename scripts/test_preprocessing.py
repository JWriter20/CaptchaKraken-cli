#!/usr/bin/env python3
"""
Script to test image preprocessing functions (greyscale and edge detection).
"""
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imagePreprocessing import to_greyscale, to_edge_outline


def main():
    parser = argparse.ArgumentParser(description="Test image preprocessing functions")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument(
        "--output-dir",
        default="test-results",
        help="Output directory for processed images (default: test-results)",
    )
    
    args = parser.parse_args()
    
    # Verify input image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = Path(args.image_path).stem
    
    # Define output paths
    greyscale_output = os.path.join(args.output_dir, f"{base_name}_greyscale.png")
    edges_output = os.path.join(args.output_dir, f"{base_name}_edges.png")
    
    # Process image
    print(f"Processing: {args.image_path}")
    print(f"Output directory: {args.output_dir}")
    
    print(f"\n1. Converting to greyscale...")
    to_greyscale(args.image_path, greyscale_output)
    print(f"   Saved: {greyscale_output}")
    
    print(f"\n2. Detecting edges...")
    to_edge_outline(args.image_path, edges_output)
    print(f"   Saved: {edges_output}")
    
    print(f"\n✓ Processing complete!")


if __name__ == "__main__":
    main()

