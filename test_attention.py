import sys
import argparse
import os
import time

# Add current directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

try:
    from src.attention import AttentionExtractor
except ImportError as e:
    print(f"Error importing AttentionExtractor: {e}")
    print("Ensure you are running this from the project root and 'src' is a package.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test SAM 3 detection and visualization")
    parser.add_argument("image", help="Path to image or video file")
    parser.add_argument("prompt", help="Text prompt for what to detect (e.g., 'a bus')")
    parser.add_argument("--output", default="test_detection.png", help="Path for output visualization")
    parser.add_argument("--max", type=int, default=10, help="Max objects to detect")
    
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    print(f"--- SAM 3 Detection Test ---")
    print(f"Target Prompt: '{args.prompt}'")
    print(f"Input Media:   {args.image}")
    
    t0 = time.time()
    print(f"Initializing AttentionExtractor...")
    extractor = AttentionExtractor()
    
    print(f"Running detection (this may take a moment to load model)...")
    try:
        # detect() returns normalized coordinates [x_min, y_min, x_max, y_max] in a dict
        detections = extractor.detect(args.image, args.prompt, max_objects=args.max)
        
        duration = time.time() - t0
        print(f"Found {len(detections)} objects in {duration:.2f}s.")
        
        for i, det in enumerate(detections):
            score = det.get('score', 0.0)
            print(f"  {i+1}: score={score:.4f}, box=[{det['x_min']:.3f}, {det['y_min']:.3f}, {det['x_max']:.3f}, {det['y_max']:.3f}]")

        if detections:
            print(f"Generating visualization: {args.output}")
            # visualize_detections handles the image loading and box drawing
            # We already fixed this method in src/attention.py to handle the detect() output format
            extractor.visualize_detections(args.image, detections, output_path=args.output)
            print(f"Success! Output saved to {args.output}")
        else:
            print("No objects detected. Try a different prompt or image.")
            
    except Exception as e:
        print(f"An error occurred during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

