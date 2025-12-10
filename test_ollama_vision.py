import ollama
import sys
import os
import time

def test_ollama_vision(image_path, model="qwen3-vl:4b"):
    print(f"Testing ollama vision with model: {model}")
    print(f"Image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found")
        return

    prompt = "What is in this image? Be brief."
    
    print("Sending request to ollama...")
    start_time = time.time()
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user", 
                "content": prompt, 
                "images": [image_path]
            }],
            options={"temperature": 0.0},
        )
        end_time = time.time()
        print(f"Response received in {end_time - start_time:.2f}s")
        print("Response content:")
        print(response["message"]["content"])
    except Exception as e:
        print(f"Error calling ollama: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ollama_vision.py <image_path> [model]")
        sys.exit(1)
    
    img = sys.argv[1]
    mdl = sys.argv[2] if len(sys.argv) > 2 else "qwen3-vl:4b"
    test_ollama_vision(img, mdl)

