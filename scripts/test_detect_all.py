import os
import sys
from PIL import Image

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from attention import AttentionExtractor

def test_detect_all():
    extractor = AttentionExtractor()
    
    test_images = [
        ("captchaimages/hcaptchaChooseSimilarShapes.png", 5),
        ("captchaimages/hcaptchaDragImage1.png", (2, 7)),
        ("captchaimages/hcaptchaDragImages3.png", (4, 6)),
        ("captchaimages/hcaptchaDragImage4.png", (4, 6)),
        ("captchaimages/hcaptchaPuzzle.png", (6, 8)),
        ("captchaimages/hcaptchaPuzzle2.png", 5),
    ]
    
    methods = ["prompts", "points"]
    
    results = {}

    for img_path, expected in test_images:
        if not os.path.exists(img_path):
            print(f"Skipping {img_path}, not found.")
            continue
            
        print(f"\nTesting {img_path} (Expected: {expected})")
        results[img_path] = {}
        
        for method in methods:
            print(f"  Running method: {method}...", end="", flush=True)
            try:
                detections = extractor.detect_all(img_path, method=method, max_objects=8)
                count = len(detections)
                results[img_path][method] = count
                
                # Check if within expected range
                if isinstance(expected, tuple):
                    passed = expected[0] <= count <= expected[1]
                else:
                    passed = count == expected
                
                status = "PASS" if passed else "FAIL"
                print(f" Found {count} objects. [{status}]")
                
                # Visualize
                output_path = f"latestDebugRun/detect_all_{os.path.basename(img_path)}_{method}.png"
                os.makedirs("latestDebugRun", exist_ok=True)
                extractor.visualize_detections(img_path, detections, output_path=output_path)
            except Exception as e:
                print(f" Error: {e}")
                results[img_path][method] = str(e)

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for img_path, res in results.items():
        print(f"{img_path}:")
        for method, count in res.items():
            print(f"  {method}: {count}")

if __name__ == "__main__":
    test_detect_all()

