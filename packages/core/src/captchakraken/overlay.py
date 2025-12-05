import base64
import io
import os
import sys
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

# Font cache to prevent repeated font loading
_FONT_CACHE = {}

_FONT_PATHS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
    '/System/Library/Fonts/Arial.ttf',
    'C:\\Windows\\Fonts\\arial.ttf',
    'arial.ttf',
    'Arial Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
]

def get_cross_platform_font(font_size: int):
    cache_key = ('system_font', font_size)
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]

    font = None
    for font_path in _FONT_PATHS:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except OSError:
            continue
    
    _FONT_CACHE[cache_key] = font
    return font

def draw_enhanced_bounding_box(draw, bbox, text, image_size):
    x1, y1, x2, y2 = bbox
    color = '#FF6B6B' # Red for buttons/interactable
    
    # Draw dashed bounding box
    dash_length = 4
    gap_length = 8
    line_width = 2

    def draw_dashed_line(start_x, start_y, end_x, end_y):
        if start_x == end_x:  # Vertical
            y = start_y
            while y < end_y:
                dash_end = min(y + dash_length, end_y)
                draw.line([(start_x, y), (start_x, dash_end)], fill=color, width=line_width)
                y += dash_length + gap_length
        else:  # Horizontal
            x = start_x
            while x < end_x:
                dash_end = min(x + dash_length, end_x)
                draw.line([(x, start_y), (dash_end, start_y)], fill=color, width=line_width)
                x += dash_length + gap_length

    draw_dashed_line(x1, y1, x2, y1)
    draw_dashed_line(x2, y1, x2, y2)
    draw_dashed_line(x2, y2, x1, y2)
    draw_dashed_line(x1, y2, x1, y1)

    if text:
        img_width, img_height = image_size
        
        # Scale font based on image width
        base_font_size = max(10, min(20, int(img_width * 0.01)))
        font = get_cross_platform_font(base_font_size)
        
        # Get text size
        if font:
            bbox_text = draw.textbbox((0, 0), text, font=font)
        else:
            bbox_text = draw.textbbox((0, 0), text)
            
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        padding = max(4, min(10, int(img_width * 0.005)))
        
        element_width = x2 - x1
        element_height = y2 - y1
        
        container_width = text_width + padding * 2
        container_height = text_height + padding * 2
        
        # Position logic
        bg_x1 = x1 + (element_width - container_width) // 2
        
        if element_width < 60 or element_height < 30:
            bg_y1 = max(0, y1 - container_height - 5)
        else:
            bg_y1 = y1 + 2
            
        bg_x2 = bg_x1 + container_width
        bg_y2 = bg_y1 + container_height
        
        # Text position
        text_x = bg_x1 + (container_width - text_width) // 2
        text_y = bg_y1 + (container_height - text_height) // 2 - bbox_text[1]
        
        # Boundary checks
        if bg_x1 < 0:
            offset = -bg_x1
            bg_x1 += offset
            bg_x2 += offset
            text_x += offset
        if bg_y1 < 0:
            offset = -bg_y1
            bg_y1 += offset
            bg_y2 += offset
            text_y += offset
        if bg_x2 > img_width:
            offset = bg_x2 - img_width
            bg_x1 -= offset
            bg_x2 -= offset
            text_x -= offset
        if bg_y2 > img_height:
            offset = bg_y2 - img_height
            bg_y1 -= offset
            bg_y2 -= offset
            text_y -= offset
            
        # Draw background and text
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color, outline='white', width=2)
        draw.text((text_x, text_y), text, fill='white', font=font)

def add_overlays_to_image(image_path: str, boxes: list[dict]):
    """
    Load an image, draw bounding boxes and numbered labels, and save it back.
    boxes: list of dicts with 'bbox': [x, y, width, height] and 'text': str
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGBA')
            draw = ImageDraw.Draw(img)
            
            for box in boxes:
                x, y, w, h = box['bbox']
                # Convert to x1, y1, x2, y2
                bbox_coords = (x, y, x + w, y + h)
                text = box['text']
                draw_enhanced_bounding_box(draw, bbox_coords, text, img.size)
                
            img = img.convert('RGB')
            img.save(image_path)
            
    except Exception as e:
        print(f"Error adding overlays: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add overlays to an image")
    parser.add_argument("image_path", help="Path to the image to modify")
    parser.add_argument("boxes_json", help="JSON string containing list of boxes with 'bbox' and 'text'")
    
    args = parser.parse_args()
    
    try:
        boxes = json.loads(args.boxes_json)
        add_overlays_to_image(args.image_path, boxes)
        print(json.dumps({"status": "success"}))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

