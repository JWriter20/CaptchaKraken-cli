import json
import os
import sys
import re
import cv2
import uuid

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.image_processor import ImageProcessor
from src.overlay import add_overlays_to_image

def generate_llava_dataset():
    input_json_path = 'captchaimages/coreRecaptcha/recaptchaAnswers.json'
    image_dir = 'captchaimages/coreRecaptcha'
    output_dir = 'captchaimages/coreRecaptcha_labeled'
    output_json_path = 'captchaimages/llava_recaptcha_dataset.json'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json_path, 'r') as f:
        answers = json.load(f)

    llava_data = []

    print(f"Processing {len(answers)} images...")

    for filename, data in answers.items():
        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            print(f"Warning: Image {filename} not found at {image_path}, skipping.")
            continue

        # Get grid boxes
        try:
            boxes = ImageProcessor.get_grid_bounding_boxes(image_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            boxes = None

        if not boxes:
            print(f"Warning: No grid detected for {filename}. Skipping overlay and LLaVA entry.")
            # We skip because without numbers, the answer key (e.g. "Tile 5") makes no sense to the model.
            continue

        # Create overlays
        overlays = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            w = x2 - x1
            h = y2 - y1
            overlays.append({
                'bbox': [x1, y1, w, h],
                'number': i + 1,
                'color': '#E74C3C' # Red
            })

        # Save overlayed image
        output_image_filename = f"labeled_{filename}"
        output_image_path = os.path.join(output_dir, output_image_filename)
        
        try:
            add_overlays_to_image(image_path, overlays, output_path=output_image_path, label_position="top-right")
        except Exception as e:
            print(f"Error saving overlay for {filename}: {e}")
            continue

        # Prepare LLaVA entry
        reasoning = data.get("reasoning", "")
        required = data.get("required", [])
        
        # Fixed prompt as requested
        human_prompt = "<image>\nSolve this grid captcha. Identify the tiles that need to be clicked. Do not include tiles that are already selected."

        # Construct GPT response
        optional = data.get("optional", [])
        to_select = sorted(list(set(required + optional)))
        
        response_data = {
            "reasoning": reasoning,
            "toSelect": to_select
        }
        gpt_response = json.dumps(response_data)

        entry = {
            "id": str(uuid.uuid4()),
            "image": output_image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": human_prompt
                },
                {
                    "from": "gpt",
                    "value": gpt_response
                }
            ]
        }
        llava_data.append(entry)

    # Save LLaVA JSON
    with open(output_json_path, 'w') as f:
        json.dump(llava_data, f, indent=2)

    print(f"Successfully generated {len(llava_data)} LLaVA entries.")
    print(f"Dataset saved to {output_json_path}")
    print(f"Labeled images saved to {output_dir}")

if __name__ == "__main__":
    generate_llava_dataset()

