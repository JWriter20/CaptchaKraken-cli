import os
import sys
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.attention import AttentionExtractor

def test_robust_detection():
    image_path = "debug_strategy2_preprocessed.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        # Try to use another image if that one is missing
        if os.path.exists("captchaimages/hcaptchaDragImage2.png"):
             image_path = "captchaimages/hcaptchaDragImage2.png"
        else:
             return

    print(f"Testing detect_robust on {image_path}...")
    
    attention = AttentionExtractor()
    
    # Prompt for something likely to be found or not found by GD
    # "The letter W" might be hard for GD if it's weirdly shaped
    prompts = ["the letter W", "white shape"]
    
    # Force GD to fail by setting high threshold, so we test Strategy 2
    detections = attention.detect_robust(
        image_path,
        prompts=prompts,
        max_objects=1,
        min_score=0.9 # High score to trigger fallback to Strategy 2
    )
    
    print(f"Result: {detections}")
    
    if detections:
        # Visualize result
        out_path = "test-results/robust_detection_result.png"
        os.makedirs("test-results", exist_ok=True)
        shutil.copy2(image_path, out_path)
        attention.visualize_detections(out_path, detections, output_path=out_path)
        print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    test_robust_detection()

