import argparse
import json
import math
import sys
from typing import Any, Dict

from PIL import Image, ImageDraw, ImageFont

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


def draw_enhanced_bounding_box(draw, bbox, text=None, number=None, image_size=None, color="#C0392B"):
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

        # Position logic - Top Left
        bg_x1 = x1 + 4
        bg_y1 = y1 + 4

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


def add_overlays_to_image(image_path: str, boxes: list[dict], output_path: str = None):
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
                    draw, bbox_coords, text=text, number=number, image_size=img.size, color=color
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
):
    """
    Add drag-and-drop visualization:
    - Red border around source
    - Arrow to target
    - Dashed border around target
    - Optional: 10% grid overlay
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGBA")
            draw = ImageDraw.Draw(img)

            # Draw source border
            # source_bbox is [x1, y1, x2, y2]
            draw_red_border(draw, source_bbox)

            if target_center:
                # Calculate source center
                source_center = ((source_bbox[0] + source_bbox[2]) / 2, (source_bbox[1] + source_bbox[3]) / 2)

                # Draw arrow
                draw_arrow(draw, source_center, target_center)

            if target_bbox:
                # Draw target dashed box (using None as text to skip label)
                draw_enhanced_bounding_box(draw, target_bbox, None, img.size)

            img = img.convert("RGB")
            img.save(image_path)

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
