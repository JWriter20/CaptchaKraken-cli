import argparse
import json
import math
import sys
import numpy as np
from typing import Any, Dict

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Font cache to prevent repeated font loading
_FONT_CACHE: Dict[tuple, Any] = {}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
    "arial.ttf",
    "Arial Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def get_cross_platform_font(font_size: int):
    cache_key = ("system_font", font_size)
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


def draw_enhanced_bounding_box(draw, bbox, text=None, number=None, image_size=None, color="#C0392B", label_position="top-left"):
    x1, y1, x2, y2 = bbox
    # color is passed as argument

    # Draw dashed bounding box
    dash_length = 8
    gap_length = 4
    line_width = 4

    def draw_dashed_line(start_x, start_y, end_x, end_y):
        # Ensure we always draw from smaller to larger coordinate
        if start_x == end_x:  # Vertical
            y_min, y_max = min(start_y, end_y), max(start_y, end_y)
            y = y_min
            while y < y_max:
                dash_end = min(y + dash_length, y_max)
                draw.line([(start_x, y), (start_x, dash_end)], fill=color, width=line_width)
                y += dash_length + gap_length
        else:  # Horizontal
            x_min, x_max = min(start_x, end_x), max(start_x, end_x)
            x = x_min
            while x < x_max:
                dash_end = min(x + dash_length, x_max)
                draw.line([(x, start_y), (dash_end, start_y)], fill=color, width=line_width)
                x += dash_length + gap_length

    draw_dashed_line(x1, y1, x2, y1)
    draw_dashed_line(x2, y1, x2, y2)
    draw_dashed_line(x2, y2, x1, y2)
    draw_dashed_line(x1, y2, x1, y1)

    label_text = ""
    if number is not None:
        label_text = str(number)
    if text:
        if label_text:
            label_text += " " + text
        else:
            label_text = text

    if label_text and image_size:
        img_width, img_height = image_size

        # Scale font based on image width - increased size for visibility
        base_font_size = max(20, min(60, int(img_width * 0.04)))
        font = get_cross_platform_font(base_font_size)

        # Get text size
        if font:
            bbox_text = draw.textbbox((0, 0), label_text, font=font)
        else:
            bbox_text = draw.textbbox((0, 0), label_text)

        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]

        padding = max(4, min(10, int(img_width * 0.005)))

        container_width = text_width + padding * 2
        container_height = text_height + padding * 2

        # Position logic
        if label_position == "center":
            # Center of the box
            box_cx = (x1 + x2) / 2
            box_cy = (y1 + y2) / 2
            
            bg_x1 = box_cx - container_width / 2
            bg_y1 = box_cy - container_height / 2
            bg_x2 = box_cx + container_width / 2
            bg_y2 = box_cy + container_height / 2
            
        elif label_position == "bottom-right":
            # Bottom Right of the box (inside)
            bg_x2 = x2 - 4
            bg_y2 = y2 - 4
            bg_x1 = bg_x2 - container_width
            bg_y1 = bg_y2 - container_height
            
        else: # Default to top-left
            bg_x1 = x1 + 4
            bg_y1 = y1 + 4
            bg_x2 = bg_x1 + container_width
            bg_y2 = bg_y1 + container_height

        # Text position
        text_x = bg_x1 + (container_width - text_width) // 2
        text_y = bg_y1 + (container_height - text_height) // 2 - bbox_text[1]

        # Boundary checks (keep label inside image)
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
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color, outline="white", width=2)
        draw.text((text_x, text_y), label_text, fill="white", font=font)


def draw_red_border(draw, bbox, width=3):
    """Draw a solid red border around a bounding box."""
    x1, y1, x2, y2 = bbox
    color = "#FF0000"
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def draw_arrow(draw, start, end, color="#FF0000", width=4):
    """Draw an arrow from start to end (high visibility for model feedback)."""
    x1, y1 = start
    x2, y2 = end

    draw.line([start, end], fill=color, width=width)

    # Arrow head
    angle = math.atan2(y2 - y1, x2 - x1)
    arrow_len = 20
    angle_offset = math.pi / 6  # 30 degrees

    p1 = (x2 - arrow_len * math.cos(angle - angle_offset), y2 - arrow_len * math.sin(angle - angle_offset))
    p2 = (x2 - arrow_len * math.cos(angle + angle_offset), y2 - arrow_len * math.sin(angle + angle_offset))

    draw.polygon([end, p1, p2], fill=color)


def add_overlays_to_image(image_path: str, boxes: list[dict], output_path: str = None, label_position="top-left"):
    """
    Load an image, draw bounding boxes and numbered labels, and save it.

    Args:
        image_path: Path to source image
        boxes: list of dicts with:
               'bbox': [x1, y1, x2, y2] (normalized) OR [x, y, w, h] (pixels)
               'text': str (optional)
               'number': int/str (optional)
               'color': str (optional)
        output_path: Path to save result (defaults to overwriting image_path)
        label_position: "top-left", "bottom-right", "center"
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGBA")
            draw = ImageDraw.Draw(img)
            width, height = img.size

            for box in boxes:
                bbox = box["bbox"]

                # Check for normalized coordinates (all <= 1.0)
                # Note: This heuristic might fail for very small images or 1x1 pixel crops,
                # but valid for standard captcha images.
                is_normalized = all(x <= 1.0 for x in bbox)

                if is_normalized:
                    # Assume [x_min, y_min, x_max, y_max] for normalized
                    x1 = bbox[0] * width
                    y1 = bbox[1] * height
                    x2 = bbox[2] * width
                    y2 = bbox[3] * height
                else:
                    # Assume [x, y, w, h] for pixels (legacy support)
                    x, y, w, h = bbox
                    x1, y1, x2, y2 = x, y, x + w, y + h

                bbox_coords = (x1, y1, x2, y2)
                text = box.get("text")
                number = box.get("number")
                color = box.get("color", "#FF6B6B")

                draw_enhanced_bounding_box(
                    draw, bbox_coords, text=text, number=number, image_size=img.size, color=color, label_position=label_position
                )

            img = img.convert("RGB")
            save_path = output_path if output_path else image_path
            img.save(save_path)

    except Exception as e:
        print(f"Error adding overlays: {e}", file=sys.stderr)
        raise


def draw_grid_overlay(draw, image_size, step=0.1):
    """Draw a 10% grid overlay with labels (more readable)."""
    width, height = image_size
    line_color = "#00FF00"  # Green
    text_color = "#00FF00"
    bg_color = (0, 32, 0, 180)  # Semi-transparent dark background for labels
    font_size = max(10, min(18, int(width * 0.025)))
    font = get_cross_platform_font(font_size)

    # Draw vertical lines and labels
    for i in range(1, int(1 / step)):
        x = int(i * step * width)
        draw.line([(x, 0), (x, height)], fill=line_color, width=1)
        label = f"{int(i * step * 100)}%"
        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
        else:
            bbox = draw.textbbox((0, 0), label)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        # Top label background
        draw.rectangle([x + 2, 2, x + 2 + w + 4, 2 + h + 4], fill=bg_color)
        draw.text((x + 4, 4), label, fill=text_color, font=font)

    # Draw horizontal lines and labels
    for i in range(1, int(1 / step)):
        y = int(i * step * height)
        draw.line([(0, y), (width, y)], fill=line_color, width=1)
        label = f"{int(i * step * 100)}%"
        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
        else:
            bbox = draw.textbbox((0, 0), label)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        # Left label background
        draw.rectangle([2, y + 2, 2 + w + 4, y + 2 + h + 4], fill=bg_color)
        draw.text((4, y + 4), label, fill=text_color, font=font)


def add_drag_overlay(
    image_path: str,
    source_bbox: list[float],
    target_bbox: list[float] = None,
    target_center: tuple = None,
    show_grid: bool = False,
    foreground_image: Image.Image = None,
):
    """
    Add drag-and-drop visualization.
    If foreground_image is provided, it uses that for the object.
    It also fills the source location in the background with a surrounding color.
    Otherwise, it uses a simple crop.
    """
    try:
        # Load image with PIL
        with Image.open(image_path) as img:
            img = img.convert("RGBA")
            
        x1, y1, x2, y2 = map(int, source_bbox)
        # Clamp
        w, h = img.size
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        
        # 1. Prepare Background
        background = img.copy()
        
        # "Remove the cropped out source image"
        # We'll simple average the color of a 5px border around the bbox and fill the rect.
        # This acts as inpainting the "hole" left by the dragged object.
        border = 5
        bx1 = max(0, x1 - border)
        by1 = max(0, y1 - border)
        bx2 = min(w, x2 + border)
        by2 = min(h, y2 + border)
        
        # Crop the border area
        region = img.crop((bx1, by1, bx2, by2))
        region_np = np.array(region)
        # Average color (ignoring alpha for now, assuming opaque bg)
        avg_color = np.mean(region_np, axis=(0, 1)).astype(int)
        fill_color = tuple(avg_color)
        
        # Draw filled rect over source
        draw_bg = ImageDraw.Draw(background)
        draw_bg.rectangle([x1, y1, x2, y2], fill=fill_color)
        
        # 2. Dim the background
        # "make it only a tad darker than the original" -> lighter dimming
        # Previously 80 alpha (approx 30%), now let's try 40 alpha (approx 15%)
        dim_overlay = Image.new("RGBA", img.size, (0, 0, 0, 40)) 
        background = Image.alpha_composite(background, dim_overlay)
        
        # 3. Prepare Object
        if foreground_image:
            # Use provided foreground image
            object_crop = foreground_image.resize((x2-x1, y2-y1)) # Ensure size matches just in case
        else:
            # Fallback to simple crop
            object_crop = img.crop((x1, y1, x2, y2))

        # 4. Paste Object at Target
        if target_center:
            tx, ty = target_center
        else:
            tx = (x1 + x2) / 2
            ty = (y1 + y2) / 2
            
        cw, ch = object_crop.size
        paste_x = int(tx - cw / 2)
        paste_y = int(ty - ch / 2)
        
        # Composite object_crop onto background
        background.alpha_composite(object_crop, (paste_x, paste_y))
        
        # 5. Highlight border (Draw with padding to avoid obscuring edges)
        draw = ImageDraw.Draw(background)
        padding = 4
        draw.rectangle(
            [paste_x - padding, paste_y - padding, paste_x + cw + padding, paste_y + ch + padding],
            outline="#00FF00",
            width=2
        )

        background = background.convert("RGB")
        background.save(image_path)

    except Exception as e:
        print(f"Error adding drag overlays: {e}", file=sys.stderr)
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
