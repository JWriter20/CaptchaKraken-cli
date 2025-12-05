#!/usr/bin/env python3
"""
Generate comprehensive HTML visualization for captcha solver testing.
Shows both planner decisions and attention extractor results.
"""

import os
import sys
import base64
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
from captchakraken.attention import AttentionExtractor
from captchakraken.planner import ActionPlanner

# Images directory
IMAGES_DIR = Path(__file__).parent / "captchaimages"
OUTPUT_HTML = Path(__file__).parent / "visualization_report.html"


@dataclass
class VisualizationResult:
    """Result from visualization test."""
    filename: str
    image_path: str
    image_width: int
    image_height: int
    image_base64: str
    
    # Planner results
    action_type: str
    target_description: str
    drag_target_description: Optional[str]
    reasoning: str
    
    # Attention results (percentages 0-1)
    click_x_pct: Optional[float] = None
    click_y_pct: Optional[float] = None
    drag_source_x_pct: Optional[float] = None
    drag_source_y_pct: Optional[float] = None
    drag_target_x_pct: Optional[float] = None
    drag_target_y_pct: Optional[float] = None
    
    # Raw text from point() calls
    source_point_query: Optional[str] = None
    target_point_query: Optional[str] = None
    
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
    <title>CaptchaKraken Visualization Report</title>
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
        
        .header .subtitle {{
            color: #888;
            font-size: 1.1rem;
        }}
        
        .stats {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1.5rem;
        }}
        
        .stat {{
            background: rgba(255,255,255,0.05);
            padding: 1rem 2rem;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            color: #4ecdc4;
        }}
        
        .stat-label {{
            font-size: 0.85rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
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
        
        .image-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }}
        
        .card-header {{
            padding: 1rem 1.5rem;
            background: linear-gradient(90deg, rgba(78, 205, 196, 0.2), rgba(69, 183, 209, 0.1));
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .card-header h3 {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1rem;
            color: #4ecdc4;
        }}
        
        .action-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 0.5rem;
        }}
        
        .action-click {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }}
        
        .action-drag {{
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: white;
        }}
        
        .action-wait {{
            background: linear-gradient(135deg, #4facfe, #00f2fe);
            color: #1a1a2e;
        }}
        
        .action-type {{
            background: linear-gradient(135deg, #a8edea, #fed6e3);
            color: #1a1a2e;
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
            margin-bottom: 1rem;
        }}
        
        .info-label {{
            font-size: 0.75rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.25rem;
        }}
        
        .info-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            color: #e0e0e0;
            background: rgba(0,0,0,0.3);
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            word-break: break-word;
        }}
        
        .coords {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }}
        
        .coord-box {{
            background: rgba(78, 205, 196, 0.1);
            border: 1px solid rgba(78, 205, 196, 0.3);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}
        
        .coord-label {{
            color: #4ecdc4;
            font-size: 0.7rem;
            text-transform: uppercase;
        }}
        
        .reasoning {{
            font-style: italic;
            color: #aaa;
            border-left: 3px solid #4ecdc4;
            padding-left: 1rem;
            margin-top: 0.5rem;
        }}
        
        .legend {{
            display: flex;
            gap: 2rem;
            justify-content: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
        }}
        
        .legend-marker {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
        }}
        
        .legend-click {{
            background: #00ff88;
            box-shadow: 0 0 10px #00ff88;
        }}
        
        .legend-drag-source {{
            background: #ff6b6b;
            box-shadow: 0 0 10px #ff6b6b;
        }}
        
        .legend-drag-target {{
            background: #4ecdc4;
            box-shadow: 0 0 10px #4ecdc4;
        }}
        
        .error-box {{
            background: rgba(255, 100, 100, 0.1);
            border: 1px solid rgba(255, 100, 100, 0.3);
            color: #ff6b6b;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🐙 CaptchaKraken Visualization Report</h1>
        <p class="subtitle">Planner + Attention Extractor Analysis</p>
        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="total-count">0</div>
                <div class="stat-label">Total Images</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="click-count">0</div>
                <div class="stat-label">Click Actions</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="drag-count">0</div>
                <div class="stat-label">Drag Actions</div>
            </div>
        </div>
    </div>
    
    <div class="legend">
        <div class="legend-item">
            <div class="legend-marker legend-click"></div>
            <span>Click Target</span>
        </div>
        <div class="legend-item">
            <div class="legend-marker legend-drag-source"></div>
            <span>Drag Source</span>
        </div>
        <div class="legend-item">
            <div class="legend-marker legend-drag-target"></div>
            <span>Drag Target</span>
        </div>
    </div>
    
    <div class="image-grid" id="image-grid"></div>
    
    <script>
        const results = {results_json};
        
        // Update stats
        document.getElementById('total-count').textContent = results.length;
        document.getElementById('click-count').textContent = results.filter(r => r.action_type === 'click').length;
        document.getElementById('drag-count').textContent = results.filter(r => r.action_type === 'drag').length;
        
        const grid = document.getElementById('image-grid');
        
        results.forEach((result, index) => {{
            const card = document.createElement('div');
            card.className = 'image-card';
            
            const actionClass = `action-${{result.action_type}}`;
            
            card.innerHTML = `
                <div class="card-header">
                    <h3>${{result.filename}}</h3>
                    <span class="action-badge ${{actionClass}}">${{result.action_type}}</span>
                </div>
                <div class="canvas-container">
                    <canvas id="canvas-${{index}}" data-index="${{index}}"></canvas>
                </div>
                <div class="info-panel">
                    ${{result.action_type === 'drag' ? `
                        <div class="info-row">
                            <div class="info-label">Source Query (Point)</div>
                            <div class="info-value">${{result.source_point_query || result.target_description}}</div>
                        </div>
                        <div class="info-row">
                            <div class="info-label">Target Query (Point)</div>
                            <div class="info-value">${{result.target_point_query || result.drag_target_description}}</div>
                        </div>
                        <div class="info-row">
                            <div class="info-label">Coordinates</div>
                            <div class="coords">
                                <div class="coord-box">
                                    <div class="coord-label">Source</div>
                                    (${{(result.drag_source_x_pct * 100).toFixed(1)}}%, ${{(result.drag_source_y_pct * 100).toFixed(1)}}%)
                                </div>
                                <div class="coord-box">
                                    <div class="coord-label">Target</div>
                                    (${{(result.drag_target_x_pct * 100).toFixed(1)}}%, ${{(result.drag_target_y_pct * 100).toFixed(1)}}%)
                                </div>
                            </div>
                        </div>
                    ` : `
                        <div class="info-row">
                            <div class="info-label">Target Query (Point)</div>
                            <div class="info-value">${{result.target_description}}</div>
                        </div>
                        <div class="info-row">
                            <div class="info-label">Coordinates</div>
                            <div class="coords">
                                <div class="coord-box">
                                    <div class="coord-label">Click</div>
                                    (${{result.click_x_pct ? (result.click_x_pct * 100).toFixed(1) : '?'}}%, ${{result.click_y_pct ? (result.click_y_pct * 100).toFixed(1) : '?'}}%)
                                </div>
                            </div>
                        </div>
                    `}}
                    <div class="info-row">
                        <div class="info-label">Planner Reasoning</div>
                        <div class="reasoning">${{result.reasoning || 'No reasoning provided'}}</div>
                    </div>
                    ${{result.error ? `<div class="error-box">${{result.error}}</div>` : ''}}
                </div>
            `;
            
            grid.appendChild(card);
        }});
        
        // Draw on canvases after images load
        results.forEach((result, index) => {{
            const canvas = document.getElementById(`canvas-${{index}}`);
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = function() {{
                // Scale canvas to fit nicely
                const maxWidth = 600;
                const scale = Math.min(1, maxWidth / img.width);
                canvas.width = img.width * scale;
                canvas.height = img.height * scale;
                
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                
                if (result.action_type === 'drag' && 
                    result.drag_source_x_pct != null && result.drag_target_x_pct != null) {{
                    // Draw drag arrow
                    const sx = result.drag_source_x_pct * canvas.width;
                    const sy = result.drag_source_y_pct * canvas.height;
                    const tx = result.drag_target_x_pct * canvas.width;
                    const ty = result.drag_target_y_pct * canvas.height;
                    
                    // Arrow line
                    ctx.beginPath();
                    ctx.moveTo(sx, sy);
                    ctx.lineTo(tx, ty);
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 4;
                    ctx.stroke();
                    
                    ctx.beginPath();
                    ctx.moveTo(sx, sy);
                    ctx.lineTo(tx, ty);
                    ctx.strokeStyle = '#ffaa00';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([8, 4]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    
                    // Arrowhead
                    const angle = Math.atan2(ty - sy, tx - sx);
                    const headLen = 20;
                    ctx.beginPath();
                    ctx.moveTo(tx, ty);
                    ctx.lineTo(tx - headLen * Math.cos(angle - Math.PI/6), ty - headLen * Math.sin(angle - Math.PI/6));
                    ctx.lineTo(tx - headLen * Math.cos(angle + Math.PI/6), ty - headLen * Math.sin(angle + Math.PI/6));
                    ctx.closePath();
                    ctx.fillStyle = '#ffaa00';
                    ctx.fill();
                    
                    // Source marker (red circle)
                    ctx.beginPath();
                    ctx.arc(sx, sy, 15, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(255, 107, 107, 0.3)';
                    ctx.fill();
                    ctx.strokeStyle = '#ff6b6b';
                    ctx.lineWidth = 3;
                    ctx.stroke();
                    
                    ctx.beginPath();
                    ctx.arc(sx, sy, 6, 0, Math.PI * 2);
                    ctx.fillStyle = '#ff6b6b';
                    ctx.fill();
                    
                    // Target marker (cyan circle)
                    ctx.beginPath();
                    ctx.arc(tx, ty, 15, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(78, 205, 196, 0.3)';
                    ctx.fill();
                    ctx.strokeStyle = '#4ecdc4';
                    ctx.lineWidth = 3;
                    ctx.stroke();
                    
                    ctx.beginPath();
                    ctx.arc(tx, ty, 6, 0, Math.PI * 2);
                    ctx.fillStyle = '#4ecdc4';
                    ctx.fill();
                    
                }} else if (result.click_x_pct != null && result.click_y_pct != null) {{
                    // Draw click marker
                    const x = result.click_x_pct * canvas.width;
                    const y = result.click_y_pct * canvas.height;
                    
                    // Crosshair
                    ctx.strokeStyle = 'rgba(0, 255, 136, 0.5)';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(canvas.width, y);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(x, 0);
                    ctx.lineTo(x, canvas.height);
                    ctx.stroke();
                    
                    // Outer glow
                    ctx.beginPath();
                    ctx.arc(x, y, 25, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(0, 255, 136, 0.2)';
                    ctx.fill();
                    
                    // Circle
                    ctx.beginPath();
                    ctx.arc(x, y, 18, 0, Math.PI * 2);
                    ctx.strokeStyle = '#00ff88';
                    ctx.lineWidth = 3;
                    ctx.stroke();
                    
                    // X marker
                    ctx.strokeStyle = '#00ff88';
                    ctx.lineWidth = 3;
                    const size = 10;
                    ctx.beginPath();
                    ctx.moveTo(x - size, y - size);
                    ctx.lineTo(x + size, y + size);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(x + size, y - size);
                    ctx.lineTo(x - size, y + size);
                    ctx.stroke();
                }}
            }};
            
            img.src = result.image_base64;
        }});
    </script>
</body>
</html>'''
    
    return html


def main():
    print("=" * 70)
    print("CaptchaKraken Visualization Generator")
    print("=" * 70)
    
    # Initialize models
    print("\n[1/3] Initializing models...")
    planner = ActionPlanner(backend="ollama")
    extractor = AttentionExtractor(backend="moondream")
    
    # Get all images
    image_files = sorted(IMAGES_DIR.glob("*.png"))
    print(f"\n[2/3] Processing {len(image_files)} images...")
    
    results = []
    
    for i, image_path in enumerate(image_files):
        filename = image_path.name
        print(f"\n{'─'*50}")
        print(f"[{i+1}/{len(image_files)}] {filename}")
        print(f"{'─'*50}")
        
        try:
            # Load image info
            img = Image.open(str(image_path))
            img_width, img_height = img.size
            img_base64 = image_to_base64(str(image_path))
            
            # Step 1: Get planner decision
            print("  → Running planner...")
            planned = planner.plan(str(image_path), "Solve this captcha")
            
            print(f"  → Action: {planned.action_type}")
            print(f"  → Target: {planned.target_description}")
            if planned.drag_target_description:
                print(f"  → Drag to: {planned.drag_target_description}")
            
            result = VisualizationResult(
                filename=filename,
                image_path=str(image_path),
                image_width=img_width,
                image_height=img_height,
                image_base64=img_base64,
                action_type=planned.action_type,
                target_description=planned.target_description or "",
                drag_target_description=planned.drag_target_description,
                reasoning=planned.reasoning or "",
            )
            
            # Step 2: Get coordinates from attention extractor
            if planned.action_type == "click":
                print(f"  → Extracting click coordinates for: {planned.target_description}")
                result.source_point_query = planned.target_description
                x_pct, y_pct = extractor.extract_coordinates(
                    str(image_path), 
                    planned.target_description
                )
                result.click_x_pct = x_pct
                result.click_y_pct = y_pct
                print(f"  → Click at: ({x_pct:.2%}, {y_pct:.2%})")
                
            elif planned.action_type == "drag":
                print(f"  → Extracting drag source: {planned.target_description}")
                result.source_point_query = planned.target_description
                sx_pct, sy_pct = extractor.extract_coordinates(
                    str(image_path), 
                    planned.target_description
                )
                result.drag_source_x_pct = sx_pct
                result.drag_source_y_pct = sy_pct
                print(f"  → Source at: ({sx_pct:.2%}, {sy_pct:.2%})")
                
                print(f"  → Extracting drag target: {planned.drag_target_description}")
                result.target_point_query = planned.drag_target_description
                tx_pct, ty_pct = extractor.extract_coordinates(
                    str(image_path), 
                    planned.drag_target_description
                )
                result.drag_target_x_pct = tx_pct
                result.drag_target_y_pct = ty_pct
                print(f"  → Target at: ({tx_pct:.2%}, {ty_pct:.2%})")
            
            results.append(result)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Still create a result with error info
            try:
                img = Image.open(str(image_path))
                results.append(VisualizationResult(
                    filename=filename,
                    image_path=str(image_path),
                    image_width=img.size[0],
                    image_height=img.size[1],
                    image_base64=image_to_base64(str(image_path)),
                    action_type="error",
                    target_description="",
                    drag_target_description=None,
                    reasoning="",
                    error=str(e)
                ))
            except:
                pass
    
    # Generate HTML
    print(f"\n[3/3] Generating HTML report...")
    html = generate_html(results)
    
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    
    print(f"\n{'='*70}")
    print(f"✓ Report saved to: {OUTPUT_HTML}")
    print(f"{'='*70}")
    
    # Summary
    clicks = sum(1 for r in results if r.action_type == "click")
    drags = sum(1 for r in results if r.action_type == "drag")
    errors = sum(1 for r in results if r.action_type == "error")
    
    print(f"\nSummary:")
    print(f"  - Total images: {len(results)}")
    print(f"  - Click actions: {clicks}")
    print(f"  - Drag actions: {drags}")
    print(f"  - Errors: {errors}")


if __name__ == "__main__":
    main()

