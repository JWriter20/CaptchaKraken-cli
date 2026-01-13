from PIL import Image
import os

img_path = "captchaimages/hcaptcha_1768163587607_3nv3u.png"
if os.path.exists(img_path):
    with Image.open(img_path) as img:
        print(f"Size: {img.size}")
        print(f"Format: {img.format}")
        print(f"Mode: {img.mode}")
else:
    print(f"File not found: {img_path}")

