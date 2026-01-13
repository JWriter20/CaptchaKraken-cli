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
        ("captchaimages/hcaptcha_1768163587607_3nv3u.png", 7),
    ]
    
    output_dir = "detect_all_prompts_only"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}

    for img_path, expected in test_images:
        if not os.path.exists(img_path):
            print(f"Skipping {img_path}, not found.")
            continue
            
        print(f"\nTesting {img_path} (Expected: {expected})")
        
        print(f"  Running detect_all (prompts method with 15% y-filter)...", end="", flush=True)
        try:
            detections = extractor.detect_all(img_path, max_objects=12)
            count = len(detections)
            results[img_path] = count
            
            print(f" Found {count} objects.")
            for i, d in enumerate(detections):
                print(f"    {i}: {d['bbox']} score={d['score']:.2f}")
            
            # Visualize
            img_name = os.path.basename(img_path).replace(".png", "")
            output_path = os.path.join(output_dir, f"{img_name}_prompts.png")
            extractor.visualize_detections(img_path, detections, output_path=output_path)
        except Exception as e:
            print(f" Error: {e}")
            results[img_path] = str(e)

    print("\n" + "="*50)
    print("SUMMARY (Prompts Method)")
    print("="*50)
    for img_path, count in results.items():
        print(f"{img_path}: {count}")

if __name__ == "__main__":
    test_detect_all()
