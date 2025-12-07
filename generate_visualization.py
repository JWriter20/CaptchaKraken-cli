#!/usr/bin/env python3
"""
Generate comprehensive HTML visualization for captcha solver testing.
Focuses on replicating the logic from tests/test_real_attention.py to visualize
AttentionExtractor performance with ideal prompts.
"""

import sys
import base64
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image, ImageDraw
from src.attention import AttentionExtractor
from src.solver import CaptchaSolver
from src.action_types import DragAction
from src.planner import ActionPlanner
from unittest.mock import MagicMock

# Images directory
IMAGES_DIR = Path(__file__).parent / "captchaimages"
OUTPUT_HTML = Path(__file__).parent / "visualization_report.html"
GROUND_TRUTH_PATH = IMAGES_DIR / "targetAreaPercentages.json"

# Test Cases from tests/test_real_attention.py
# Format: (filename, method, prompt, valid_region_keys, destination_key, source_prompt, dest_prompt)
TEST_CASES = [
    # Checkboxes
    ("cloudflare.png", "focus", "checkbox", ["checkbox"], None, None, None),
    ("hcaptchaBasic.png", "focus", "checkbox", ["checkbox"], None, None, None),
    ("recaptchaBasic.png", "focus", "checkbox", ["checkbox"], None, None, None),
    
    # Image Selection
    ("hcaptchaImages1.png", "detect", "bird", ["topMiddleBirdGroup", "topRightBirdGroup", "bottomLeftBirdGroup"], None, None, None),
    ("recaptchaImages.png", "detect", "car", ["promptCar", "topMiddleCarGroup", "middleRightCarGroup", "bottomRightCar"], None, None, None),
    ("recaptchaImages2.png", "detect", "motorcycle", ["motorcycleContainer"], None, None, None),
    ("recaptchaImages3.png", "detect", "fire hydrant", ["bottomLeftFireHydrant", "middleFireHydrant", "topRightFireHydrant"], None, None, None),
    
    # Drag Puzzles - Source
    ("hcaptchaDragImage1.png", "detect", "top segment movable square", ["topSegment"], None, None, None),
    ("hcaptchaDragImage1.png", "detect", "empty puzzle slot", ["topSegmentDesination"], None, None, None),
    ("hcaptchaDragImage1.png", "detect", "dark shadow hole", ["topSegmentDesination"], None, None, None),
    ("hcaptchaDragImage1.png", "detect", "missing piece area", ["topSegmentDesination"], None, None, None),
    ("hcaptchaDragImage1.png", "detect", "dark cutout on the left", ["topSegmentDesination"], None, None, None),
    ("hcaptchaDragImage2.png", "detect", "top movable deer head", ["deerhead"], None, None, None),
    ("hcaptchaDragImages3.png", "detect", "bottom right movable bee", ["bee"], None, None, None),
    
    # Drag Puzzles - Destination
    ("hcaptchaDragImages3.png", "detect", "top left strawberry", ["beeDesinationStrawberry"], None, None, None),
    
    # Drag Puzzles - Iterative Solver
    # Added detection_prompt (source) and dest_prompt (target)
    ("hcaptchaDragImage2.png", "iterative_drag", "Drag the deer head to the body", ["deerhead"], "deerheadDesination", "top movable deer head", None),
    ("hcaptchaDragImage1.png", "iterative_drag", "Drag the puzzle piece to the dark cutout on the left", ["topSegment"], "topSegmentDesination", "top segment movable square", "dark cutout on the left"),
    
    # Drag Puzzles - One-shot Detect
    ("hcaptchaDragImages3.png", "detect_drag_destination", "Drag the bee to the strawberry", ["bee"], "beeDesinationStrawberry", "bottom right movable bee", None),
    ("hcaptchaDragImage1.png", "detect_drag_destination", "Drag the puzzle piece to the empty slot", ["topSegment"], "topSegmentDesination", "top segment movable square", None),
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
    drag_steps: Optional[List[dict]] = None  # List of {step, image_base64, description} for iterative
    
    # Error info
    error: Optional[str] = None


def image_to_base64(image_path: str) -> str:
    """Convert image to base64 data URI."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = Path(image_path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def load_ground_truth(filename: str, key: str) -> Optional[List[float]]:
    """Load ground truth bbox for a given file and key."""
    if not GROUND_TRUTH_PATH.exists():
        return None
    
    try:
        with open(GROUND_TRUTH_PATH, 'r') as f:
            data = json.load(f)
        
        file_data = None
        for item in data:
            if filename in item:
                file_data = item[filename]
                break
        
        if not file_data:
            return None
            
        regions = file_data.get("target_bounding_boxes") or file_data.get("target_area_percentages")
        if not regions:
            return None
            
        for region_map in regions:
            if key in region_map:
                return region_map[key]
                
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None
    return None

def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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
            
            card.id = `card-${{index}}`;
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
                    
                    if (result.drag_steps) {{
                        // For iterative drag, show steps below the main image
                        const stepsContainer = document.createElement('div');
                        stepsContainer.style.display = 'grid';
                        stepsContainer.style.gridTemplateColumns = 'repeat(3, 1fr)';
                        stepsContainer.style.gap = '10px';
                        stepsContainer.style.marginTop = '10px';
                        stepsContainer.style.padding = '10px';
                        
                        result.drag_steps.forEach(step => {{
                            const stepDiv = document.createElement('div');
                            stepDiv.innerHTML = `
                                <img src="${{step.image_base64}}" style="width: 100%; border-radius: 4px;">
                                <div style="font-size: 0.8rem; color: #888; margin-top: 4px;">${{step.step}}</div>
                                <div style="font-size: 0.7rem; color: #666;">${{step.description}}</div>
                            `;
                            stepsContainer.appendChild(stepDiv);
                        }});
                        
                        // Append to the info panel of the card
                        const infoPanel = document.querySelector(`#card-${{index}} .info-panel`);
                        if (infoPanel) infoPanel.appendChild(stepsContainer);
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", help="Filter test cases by string matching filename or method")
    parser.add_argument("--drag-only", action="store_true", help="Run only drag tests")
    args = parser.parse_args()

    print("=" * 70)
    print("Attention Visualization Generator")
    print("=" * 70)
    
    print("\n[1/3] Initializing AttentionExtractor...")
    extractor = AttentionExtractor(backend="moondream")
    print("\n[1/3] Initializing ActionPlanner for real drag queries...")
    planner = ActionPlanner()
    
    print(f"\n[2/3] Processing {len(TEST_CASES)} test cases...")
    results = []
    
    for i, (filename, method, prompt, valid_keys, dest_key, detection_prompt, dest_prompt) in enumerate(TEST_CASES):
        if args.drag_only and "drag" not in method and "drag" not in filename.lower():
            continue
        if args.filter and (args.filter not in filename and args.filter not in method):
            continue

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
                # Use explicit detection_prompt if provided, else use prompt
                detect_prompt = detection_prompt if detection_prompt else prompt
                detections = extractor.detect_objects(str(image_path), detect_prompt)
                boxes = [d['bbox'] for d in detections]
                result.bounding_boxes = boxes
                print(f"  → Found {len(boxes)} detections")
                
                # Check accuracy against valid_keys if available
                if boxes and valid_keys:
                    # Try to find GT for the first valid key
                    key = valid_keys[0]
                    ground_truth_bbox = load_ground_truth(filename, key)
                    if ground_truth_bbox:
                         gt_cx = (ground_truth_bbox[0] + ground_truth_bbox[2]) / 2
                         gt_cy = (ground_truth_bbox[1] + ground_truth_bbox[3]) / 2
                         
                         best_box = boxes[0] 
                         cand_cx = (best_box[0] + best_box[2]) / 2
                         cand_cy = (best_box[1] + best_box[3]) / 2
                         
                         dist = math.sqrt((cand_cx - gt_cx)**2 + (cand_cy - gt_cy)**2)
                         print(f"  → Distance to GT ('{key}'): {dist:.3f}")
                
            elif method == "detect_drag_destination":
                print("  → Running one-shot detection logic")
                
                # 1. Ask planner for the target description
                strategy = planner.plan_drag_strategy(str(image_path), prompt)
                target_description = strategy.get("destination_prompt")
                print(f"  → Planner identified target: '{target_description}'")
                
                # 2. Detect that target
                detections = extractor.detect_objects(str(image_path), target_description)
                boxes = [d['bbox'] for d in detections]
                result.bounding_boxes = boxes
                print(f"  → Found {len(boxes)} destination candidates")
                
                if boxes:
                    # Visualization: Show detection on image
                    # We can reuse the standard bbox drawing in the HTML
                    # But we also want to calculate error if ground truth exists
                    
                    # 0. Get ground truth if available
                    ground_truth_bbox = load_ground_truth(filename, dest_key) if dest_key else None
                    if ground_truth_bbox:
                         gt_cx = (ground_truth_bbox[0] + ground_truth_bbox[2]) / 2
                         gt_cy = (ground_truth_bbox[1] + ground_truth_bbox[3]) / 2
                         print(f"  → Ground Truth Center: ({gt_cx:.2f}, {gt_cy:.2f})")
                         
                         # Best candidate is closest to center or highest confidence (assumed 1st is best)
                         best_box = boxes[0] 
                         cand_cx = (best_box[0] + best_box[2]) / 2
                         cand_cy = (best_box[1] + best_box[3]) / 2
                         
                         dist = math.sqrt((cand_cx - gt_cx)**2 + (cand_cy - gt_cy)**2)
                         print(f"  → Distance to GT: {dist:.3f}")

            elif method == "iterative_drag":
                print("  → Running iterative drag visualization (real planner)")
                
                from src.overlay import add_drag_overlay
                import shutil
                import tempfile
                import os
                
                # 0. Get ground truth if available
                ground_truth_bbox = load_ground_truth(filename, dest_key) if dest_key else None
                ground_truth_center = None
                if ground_truth_bbox:
                    gt_cx = (ground_truth_bbox[0] + ground_truth_bbox[2]) / 2
                    gt_cy = (ground_truth_bbox[1] + ground_truth_bbox[3]) / 2
                    ground_truth_center = (gt_cx, gt_cy)
                    print(f"  → Ground Truth Center: ({gt_cx:.2f}, {gt_cy:.2f})")
                
                # 1. Detect source
                # Use explicit detection_prompt if provided, else use prompt
                detect_prompt = detection_prompt if detection_prompt else prompt
                print(f"  → Detecting source with prompt: '{detect_prompt}'")
                
                detections = extractor.detect_objects(str(image_path), detect_prompt)
                if not detections:
                    result.error = "Could not find draggable"
                    results.append(result)
                    continue
                    
                source_bbox_pct = detections[0]['bbox']
                source_center_pct = (
                    (source_bbox_pct[0] + source_bbox_pct[2]) / 2,
                    (source_bbox_pct[1] + source_bbox_pct[3]) / 2,
                )
                
                # Convert to px
                source_bbox_px = [
                    source_bbox_pct[0] * width,
                    source_bbox_pct[1] * height,
                    source_bbox_pct[2] * width,
                    source_bbox_pct[3] * height
                ]
                
                drag_steps = []
                
                # Step 0: Original found
                drag_steps.append({
                    "step": "Detection", 
                    "image_base64": b64_img,
                    "description": f"Found draggable at {source_bbox_pct}"
                })
                
                # Step 1: Initial Overlay (Source only)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    step1_path = tf.name
                shutil.copy2(str(image_path), step1_path)
                
                # If ground truth exists, maybe draw it lightly? 
                # Currently add_drag_overlay doesn't support extra arbitrary boxes easily without modification
                # but we can modify the image before calling add_drag_overlay if we really wanted to.
                # For now, we rely on the description.
                
                add_drag_overlay(step1_path, source_bbox_px, show_grid=True)
                drag_steps.append({
                    "step": "1. Identify Source",
                    "image_base64": image_to_base64(step1_path),
                    "description": "Highlighted source for model query"
                })
                
                # Step 2: Ask planner for destination using overlay image (matches solver's iterative flow)
                # OR use explicit detection if provided
                initial_plan = {}
                if dest_prompt:
                    print(f"  → Initializing with detection: '{dest_prompt}'")
                    dest_detections = extractor.detect_objects(str(image_path), dest_prompt)
                    if dest_detections:
                        d_bbox = dest_detections[0]['bbox']
                        cx = (d_bbox[0] + d_bbox[2]) / 2
                        cy = (d_bbox[1] + d_bbox[3]) / 2
                        initial_plan = {"target_x": cx, "target_y": cy, "reasoning": f"Detected {dest_prompt}"}
                        print(f"  → Detected target at {cx:.2f}, {cy:.2f}")
                
                if not initial_plan:
                    initial_plan = planner.get_drag_destination(
                        step1_path,
                        prompt_text=prompt,
                        draggable_bbox_pct=source_bbox_pct,
                    )
                
                os.unlink(step1_path)
                current_target_pct = [
                    initial_plan.get("target_x", source_center_pct[0]),
                    initial_plan.get("target_y", source_center_pct[1]),
                ]
                if any(v is None for v in current_target_pct):
                    current_target_pct = [source_center_pct[0], source_center_pct[1]]
                
                # Iteratively refine until model marks the move as correct
                max_iterations = 3
                for iteration in range(max_iterations):
                    # Clamp and build bbox for this iteration
                    current_target_pct = [
                        max(0.0, min(1.0, current_target_pct[0])),
                        max(0.0, min(1.0, current_target_pct[1])),
                    ]
                    w_pct = source_bbox_pct[2] - source_bbox_pct[0]
                    h_pct = source_bbox_pct[3] - source_bbox_pct[1]
                    target_bbox_pct = [
                        max(0.0, current_target_pct[0] - w_pct / 2),
                        max(0.0, current_target_pct[1] - h_pct / 2),
                        min(1.0, current_target_pct[0] + w_pct / 2),
                        min(1.0, current_target_pct[1] + h_pct / 2),
                    ]
                    target_center_px = (current_target_pct[0] * width, current_target_pct[1] * height)
                    target_bbox_px = [
                        target_bbox_pct[0] * width,
                        target_bbox_pct[1] * height,
                        target_bbox_pct[2] * width,
                        target_bbox_pct[3] * height,
                    ]
                    
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                        model_path = tf.name
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                        vis_path = tf.name
                        
                    shutil.copy2(str(image_path), model_path)
                    shutil.copy2(str(image_path), vis_path)
                    
                    # If we have ground truth, draw it in Green on VISUALIZATION ONLY
                    if ground_truth_bbox:
                         with Image.open(vis_path) as tmp_img:
                            tmp_img = tmp_img.convert('RGBA')
                            tmp_draw = ImageDraw.Draw(tmp_img)
                            gt_px = [
                                ground_truth_bbox[0] * width,
                                ground_truth_bbox[1] * height,
                                ground_truth_bbox[2] * width,
                                ground_truth_bbox[3] * height
                            ]
                            tmp_draw.rectangle(gt_px, outline='#00FF00', width=3)
                            tmp_img = tmp_img.convert('RGB')
                            tmp_img.save(vis_path)

                    # Add overlay to both (model gets standard overlay, vis gets overlay + GT)
                    add_drag_overlay(
                        model_path, 
                        source_bbox_px, 
                        target_bbox=target_bbox_px, 
                        target_center=target_center_px,
                        show_grid=True
                    )
                    add_drag_overlay(
                        vis_path, 
                        source_bbox_px, 
                        target_bbox=target_bbox_px, 
                        target_center=target_center_px,
                        show_grid=True
                    )
                    
                    dx_pct = (current_target_pct[0] - source_center_pct[0]) * 100
                    dy_pct = (current_target_pct[1] - source_center_pct[1]) * 100
                    dx_text = f"{dx_pct:+.1f}%"
                    dy_text = f"{dy_pct:+.1f}%"
                    
                    # Calculate error if GT available
                    error_text = ""
                    dist = 0
                    if ground_truth_center:
                        dist = math.sqrt(
                            (current_target_pct[0] - ground_truth_center[0])**2 + 
                            (current_target_pct[1] - ground_truth_center[1])**2
                        )
                        error_text = f" | Error: {dist*100:.1f}%"
                    
                    refinement = planner.refine_drag(model_path, prompt_text=prompt)
                    status = refinement.get("status", "needs_adjustment")
                    adjustment = refinement.get("adjustment", {}) or {}
                    adj_x = adjustment.get("x_offset", 0) or 0
                    adj_y = adjustment.get("y_offset", 0) or 0
                    reasoning = refinement.get("reasoning", "")
                    
                    description = (
                        f"Iter {iteration + 1}: Move {dx_text} horizontally, {dy_text} vertically "
                        f"(target {current_target_pct}){error_text}"
                    )
                    if iteration == 0 and initial_plan.get("reasoning"):
                        description += f" | {initial_plan.get('reasoning','')}"
                    
                    if status == "correct":
                        description += " | Model accepted target"
                    else:
                        description += (
                            f" | Feedback: {status}, adjust x {adj_x:+.2f}, y {adj_y:+.2f} {reasoning}"
                        )
                    
                    print(f"    {description}") # Print to stdout for debugging
                    
                    drag_steps.append({
                        "step": f"2.{iteration + 1} Iteration",
                        "image_base64": image_to_base64(vis_path),
                        "description": description
                    })
                    os.unlink(model_path)
                    os.unlink(vis_path)
                    
                    if status == "correct":
                        # If we have ground truth and error is high, note it
                        if ground_truth_center and dist > 0.05: # 5% tolerance
                             drag_steps[-1]["description"] += " | ⚠️ ACCEPTED BUT WRONG (>5%)"
                        elif ground_truth_center and dist <= 0.05:
                             drag_steps[-1]["description"] += " | ✅ SUCCESS"
                        break
                    
                    # Apply adjustment for next iteration
                    current_target_pct = [
                        current_target_pct[0] + adj_x,
                        current_target_pct[1] + adj_y,
                    ]
                    
                    # Stop if adjustments are negligible
                    if abs(adj_x) < 0.01 and abs(adj_y) < 0.01:
                        drag_steps[-1]["description"] += " | Adjustment tiny, stopping"
                        if ground_truth_center and dist > 0.05:
                            drag_steps[-1]["description"] += " | ⚠️ STOPPED BUT WRONG"
                        break
                
                result.drag_steps = drag_steps

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
