import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.captchakraken.parser import CaptchaParser

def test_parsing():
    image_path = "captchaimages/cloudflare.png"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print(f"Testing parsing on {image_path}...")
    try:
        parser = CaptchaParser()
        components, labeled_image_b64 = parser.parse(image_path)
        
        print(f"Successfully parsed {len(components)} components:")
        for comp in components:
            print(f"- ID {comp.id}: {comp.label} ({comp.type}) Box: {comp.box}")
            
    except Exception as e:
        print(f"Parsing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parsing()

