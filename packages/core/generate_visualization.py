#!/usr/bin/env python3
"""
Generate comprehensive HTML visualization for captcha solver testing.
Focuses on replicating the logic from tests/test_real_attention.py to visualize
AttentionExtractor performance with ideal prompts.
"""

import sys
import base64
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
from captchakraken.attention import AttentionExtractor

# Images directory
IMAGES_DIR = Path(__file__).parent / "captchaimages"
OUTPUT_HTML = Path(__file__).parent / "visualization_report.html"

# Test Cases from tests/test_real_attention.py
# Format: (filename, method, prompt, valid_region_keys)
TEST_CASES = [
    # Checkboxes
    ("cloudflare.png", "focus", "checkbox", ["checkbox"]),
    ("hcaptchaBasic.png", "focus", "checkbox", ["checkbox"]),
    ("recaptchaBasic.png", "focus", "checkbox", ["checkbox"]),
    
    # Image Selection
    ("hcaptchaImages1.png", "detect", "bird", ["topMiddleBirdGroup", "topRightBirdGroup", "bottomLeftBirdGroup"]),
    ("recaptchaImages.png", "detect", "car", ["promptCar", "topMiddleCarGroup", "middleRightCarGroup", "bottomRightCar"]),
    ("recaptchaImages2.png", "detect", "motorcycle", ["motorcycleContainer"]),
    ("recaptchaImages3.png", "detect", "fire hydrant", ["bottomLeftFireHydrant", "middleFireHydrant", "topRightFireHydrant"]),
    
    # Drag Puzzles - Source
    ("hcaptchaDragImage1.png", "detect", "top segment movable square", ["topSegment"]),
    ("hcaptchaDragImage2.png", "detect", "top movable deer head", ["deerhead"]),
    ("hcaptchaDragImages3.png", "detect", "bottom right movable bee", ["bee"]),
    
    # Drag Puzzles - Destination
    ("hcaptchaDragImages3.png", "detect", "top left strawberry", ["beeDesinationStrawberry"]),
]

@dataclass
class VisualizationResult:
    """Result from visualization test."""
    filename: str
    image_path: str
    image_width: int
    image_height: int
    image_base64: str
    prompt_text: str
    method: str
    
    # Results
    focus_point: Optional[Tuple[float, float]] = None
    bounding_boxes: Optional[List[List[float]]] = None
    
    # Error info
    error: Optional[str] = None


def image_to_base64(image_path: str) -> str:
    """Convert image to base64 data URI."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = Path(image_path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def generate_html(results: List[VisualizationResult]) -> str:
    """Generate the HTML visualization page."""
    
    results_json = json.dumps([asdict(r) for r in results])
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CaptchaKraken Attention Visualization</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Space Grotesk', sans-serif;
            background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 2rem;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, rgba(255,100,100,0.1) 0%, rgba(100,100,255,0.1) 100%);
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #45b7d1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }}
        
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
        }}
        
        .image-card {{
            background: rgba(20, 20, 35, 0.8);
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .card-header {{
            padding: 1rem 1.5rem;
            background: linear-gradient(90deg, rgba(78, 205, 196, 0.2), rgba(69, 183, 209, 0.1));
            border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .method-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            background: rgba(255, 255, 255, 0.1);
        }}
        
        .canvas-container {{
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #0a0a0f;
            padding: 1rem;
        }}
        
        .canvas-container canvas {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}
        
        .info-panel {{
            padding: 1.5rem;
        }}
        
        .info-row {{
            margin-bottom: 0.5rem;
        }}
        
        .info-label {{
            font-size: 0.75rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .info-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            color: #4ecdc4;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>👁️ Attention Extractor Analysis</h1>
        <p>Visualizing test_real_attention.py results</p>
    </div>
    
    <div class="image-grid" id="image-grid"></div>
    
    <script>
        const results = {results_json};
        const grid = document.getElementById('image-grid');
        
        results.forEach((result, index) => {{
            const card = document.createElement('div');
            card.className = 'image-card';
            
            card.innerHTML = `
                <div class="card-header">
                    <h3>${{result.filename}}</h3>
                    <span class="method-badge">${{result.method}}</span>
                </div>
                <div class="canvas-container">
                    <canvas id="canvas-${{index}}"></canvas>
                </div>
                <div class="info-panel">
                    <div class="info-row">
                        <div class="info-label">Prompt</div>
                        <div class="info-value">${{result.prompt_text}}</div>
                    </div>
                    ${{result.error ? `<div class="info-row" style="color: #ff6b6b">${{result.error}}</div>` : ''}}
                </div>
            `;
            
            grid.appendChild(card);
            
            // Draw on canvas
            setTimeout(() => {{
                const canvas = document.getElementById(`canvas-${{index}}`);
                const ctx = canvas.getContext('2d');
                const img = new Image();
                
                img.onload = () => {{
                    // Scale for display
                    const maxWidth = 800;
                    const scale = Math.min(1, maxWidth / img.width);
                    canvas.width = img.width * scale;
                    canvas.height = img.height * scale;
                    
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    
                    // Draw results
                    if (result.focus_point) {{
                        const [x, y] = result.focus_point;
                        const px = x * canvas.width;
                        const py = y * canvas.height;
                        
                        // Crosshair
                        ctx.strokeStyle = '#00ff88';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(px - 10, py);
                        ctx.lineTo(px + 10, py);
                        ctx.moveTo(px, py - 10);
                        ctx.lineTo(px, py + 10);
                        ctx.stroke();
                        
                        // Circle
                        ctx.beginPath();
                        ctx.arc(px, py, 5, 0, Math.PI * 2);
                        ctx.fillStyle = '#00ff88';
                        ctx.fill();
                    }}
                    
                    if (result.bounding_boxes) {{
                        result.bounding_boxes.forEach(bbox => {{
                            // bbox is [x_min, y_min, x_max, y_max] normalized
                            const x1 = bbox[0] * canvas.width;
                            const y1 = bbox[1] * canvas.height;
                            const w = (bbox[2] - bbox[0]) * canvas.width;
                            const h = (bbox[3] - bbox[1]) * canvas.height;
                            
                            ctx.strokeStyle = '#ff6b6b';
                            ctx.lineWidth = 3;
                            ctx.strokeRect(x1, y1, w, h);
                            
                            ctx.fillStyle = 'rgba(255, 107, 107, 0.2)';
                            ctx.fillRect(x1, y1, w, h);
                        }});
                    }}
                }};
                
                img.src = result.image_base64;
            }}, 0);
        }});
    </script>
</body>
</html>'''
    
    return html


def main():
    print("=" * 70)
    print("Attention Visualization Generator")
    print("=" * 70)
    
    print("\n[1/3] Initializing AttentionExtractor...")
    extractor = AttentionExtractor(backend="moondream")
    
    print(f"\n[2/3] Processing {len(TEST_CASES)} test cases...")
    results = []
    
    for i, (filename, method, prompt, valid_keys) in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] {filename} ({method})")
        print(f"  Prompt: '{prompt}'")
        
        image_path = IMAGES_DIR / filename
        if not image_path.exists():
            print(f"  ✗ Image not found: {image_path}")
            continue
            
        try:
            img = Image.open(str(image_path))
            width, height = img.size
            b64_img = image_to_base64(str(image_path))
            
            result = VisualizationResult(
                filename=filename,
                image_path=str(image_path),
                image_width=width,
                image_height=height,
                image_base64=b64_img,
                prompt_text=prompt,
                method=method
            )
            
            if method == "focus":
                point = extractor.focus(str(image_path), prompt)
                result.focus_point = point
                print(f"  → Point: {point}")
                
            elif method == "detect":
                detections = extractor.detect_objects(str(image_path), prompt)
                boxes = [d['bbox'] for d in detections]
                result.bounding_boxes = boxes
                print(f"  → Found {len(boxes)} detections")
                
            results.append(result)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\n[3/3] Generating HTML report...")
    html = generate_html(results)
    
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
        
    print(f"\n{'='*70}")
    print(f"✓ Report saved to: {OUTPUT_HTML}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
