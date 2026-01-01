import requests
import base64
import time
import os

image_path = "captchaimages/hcaptchaPuzzle2.png"
with open(image_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

prompt = """Analyze the captcha and solve it using direct actions or tool calls.

Favor direct actions with descriptions over tool calls when possible.
Please solve this hCaptcha puzzle.
Direct Actions:
1. { "action": "click", "target_description": "ID or description", "max_items": N }
   - Use this to click one or more objects.
   - target_description: Preferred.
   - max_items: Number of matches to click (default 1).
2. { "action": "drag", "source_description": "ID or desc", "target_description": "desc of destination", "location_hint": [x, y] }
   - Use this for drag-and-drop puzzles.
}"""

print("Testing /api/generate...")
t0 = time.time()
resp = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "test-qwen3-vl:latest",
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "format": "json"
    }
)
t1 = time.time()
print(f"Time: {t1-t0:.2f}s")
data = resp.json()
print(f"Total Duration (from stats): {data.get('total_duration', 0)/1e9:.2f}s")
print(f"Prompt Eval Count: {data.get('prompt_eval_count')}")
print(f"Eval Count: {data.get('eval_count')}")

print("\nTesting /api/chat...")
t0 = time.time()
resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "test-qwen3-vl:latest",
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
        "format": "json"
    }
)
t1 = time.time()
print(f"Time: {t1-t0:.2f}s")
data = resp.json()
print(f"Total Duration (from stats): {data.get('total_duration', 0)/1e9:.2f}s")
