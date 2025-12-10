
import sys
import os
import cv2
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from attention import AttentionExtractor
from planner import ActionPlanner

def test_w_extraction():
    extractor = AttentionExtractor()
    image_path = "captchaimages/sourceWImage.png"
    output_debug_path = "debug_w_color_k3_merged.png"
    
    print(f"Testing W extraction on {image_path}...")
    
    # 1. Run Color Segmentation (we know k=3, merge=True works well)
    print("Running segmentation...")
    result = extractor.segment_by_color(image_path, k=3, merge_components=True)
    masks = result.get("masks", [])
    
    if not masks:
        print("Error: No masks found!")
        sys.exit(1)
        
    print(f"Found {len(masks)} masks.")
    
    # 2. Save debug image for Planner (or just reuse the one we know exists if we want to skip step 1 in valid test? 
    # No, let's generate it to be sure the pipeline works)
    extractor.visualize_masks(image_path, masks, output_path=output_debug_path, draw_ids=True)
    
    # 3. Use Planner to select "the letter W"
    # Note: This requires an API key. If not present, we might mock or fallback.
    # For now, let's assume we can try. If it fails (no key), we fallback to the hardcoded expectation for this specific test case.
    
    selected_ids = []
    
    try:
        planner = ActionPlanner()
        # Mocking the prompt slightly to be very specific
        target_description = "the letter W"
        print(f"Asking planner to find '{target_description}'...")
        
        # We need to handle if the environment doesn't have the key set up
        if not planner.gemini_api_key and not os.getenv("GEMINI_API_KEY"):
             raise ValueError("No Gemini API Key found")

        selected_ids = planner.select_items(output_debug_path, target_description, len(masks))
        print(f"Planner selected IDs: {selected_ids}")
        
    except Exception as e:
        print(f"Planner failed (expected if no API key): {e}")
        print("Falling back to center-most mask.")
        
        # Heuristic: Find mask closest to center
        import cv2
        best_idx = 0
        min_dist = float('inf')
        
        # Load image to get dims
        with Image.open(image_path) as tmp_img:
            w, h = tmp_img.size
            center = (w // 2, h // 2)
            
        for idx, m in enumerate(masks):
            if len(m.shape) == 3: m = m[0]
            M = cv2.moments(m.astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                dist = (cX - center[0])**2 + (cY - center[1])**2
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
        
        selected_ids = [best_idx]
        print(f"Fallback selected ID: {best_idx}")

    if not selected_ids:
        print("No selection made.")
        sys.exit(1)
        
    # 4. Clip the image using the selected mask
    # We'll use the first selected ID
    target_id = selected_ids[0] # This ID refers to the label in the image
    
    # In visualize_masks, labels are the original indices.
    # So we can access masks directly by this ID.
    
    if target_id >= len(masks):
        print(f"Error: Selected ID {target_id} out of range.")
        sys.exit(1)
        
    target_mask = masks[target_id]
    if len(target_mask.shape) == 3: target_mask = target_mask[0]
    
    # Apply mask to original image
    print("Applying mask to clip image...")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) # Add alpha channel
    
    # Create alpha mask from our binary mask
    # target_mask is boolean or 0/255. 
    # Ensure it's 0/1
    binary_mask = (target_mask > 0).astype(np.uint8)
    
    # Apply to alpha channel
    # Where mask is 0, alpha should be 0 (transparent)
    # Where mask is 1, alpha should be 255 (opaque)
    img[:, :, 3] = binary_mask * 255
    
    # Save result
    output_clipped = "clipped_w_result.png"
    cv2.imwrite(output_clipped, img)
    print(f"Saved clipped result to {output_clipped}")
    
    # Optional: Save a version with green background to verify
    img_green = cv2.imread(image_path)
    img_green[binary_mask == 0] = [0, 255, 0] # Replace background with green
    cv2.imwrite("clipped_w_on_green.png", img_green)


if __name__ == "__main__":
    test_w_extraction()

