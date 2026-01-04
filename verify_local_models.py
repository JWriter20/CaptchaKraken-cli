
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoModel
from qwen_vl_utils import process_vision_info
import os
import time
import sys
from PIL import Image
from src.attention import AttentionExtractor

# Add src to path just in case
sys.path.append(os.getcwd())

def test_model(model_id, image_path):
    print(f"\n" + "="*50)
    print(f"Testing model: {model_id}")
    print(f"="*50)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        print(f"Loading {model_id} on {device} with {dtype}...")
        t0 = time.time()
        # Use AutoModelForImageTextToText with trust_remote_code
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        t1 = time.time()
        print(f"Model loaded in {t1-t0:.2f}s")
        
        instruction = "Solve the captcha grid. Identify the tiles that need to be clicked."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        print("Running inference...")
        t2 = time.time()
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        t3 = time.time()
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print(f"Inference took {t3-t2:.2f}s")
        print(f"Output: {output_text[0]}")
        
        # Cleanup to free VRAM for next model
        del model
        del processor
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True
    except Exception as e:
        print(f"Error testing {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sam3(image_path):
    print(f"\n" + "="*50)
    print(f"Testing SAM3 (Jake-Writer-Jobharvest/sam3)")
    print(f"="*50)
    
    try:
        extractor = AttentionExtractor()
        print("Loading SAM3...")
        extractor.load_models()
        
        print(f"Running detection for 'objects' in {image_path}...")
        t0 = time.time()
        results = extractor.detect(image_path, "objects", max_objects=5)
        t1 = time.time()
        
        print(f"SAM3 detection took {t1-t0:.2f}s")
        print(f"Found {len(results)} objects")
        for i, res in enumerate(results):
            print(f"  Object {i+1}: {res}")
            
        return True
    except Exception as e:
        print(f"Error testing SAM3: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_image = "captchaimages/hcaptchaDragImage1.png"
    if not os.path.exists(test_image):
        print(f"Error: Test image {test_image} not found.")
        sys.exit(1)
        
    print(f"Starting verification with image: {test_image}")
    
    # 1. Test SAM3
    sam_ok = test_sam3(test_image)
    
    # 2. Test BF16 Merged
    BF16_ok = test_model("Jake-Writer-Jobharvest/qwen3-vl-8b-merged-BF16", test_image)
    
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print(f"SAM3: {'PASS' if sam_ok else 'FAIL'}")
    print(f"Qwen3-VL BF16 Merged: {'PASS' if BF16_ok else 'FAIL'}")
    print("="*50)
