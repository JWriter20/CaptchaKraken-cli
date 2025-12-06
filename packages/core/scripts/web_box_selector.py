"""Lightweight web UI to draw a bounding box over a captcha image.

Run (from repo root):
    python packages/core/scripts/web_box_selector.py packages/core/captchaimages/cloudflare.png

Then open the printed URL in your browser (Cursor browser works over SSH).
Draw/resize the box; the normalized (x1, y1, x2, y2) in [0,1] updates live and
can be copied via the button.
"""

from __future__ import annotations

import argparse
import http.server
import os
import shutil
import socketserver
import tempfile
from pathlib import Path

from PIL import Image


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Box Selector</title>
  <style>
    body { margin: 0; padding: 0; font-family: sans-serif; background: #111; color: #eee; }
    .container { position: relative; display: inline-block; margin: 16px; }
    img { display: block; max-width: 90vw; max-height: 80vh; border: 1px solid #555; user-select: none; -webkit-user-drag: none; }
    #overlay { position: absolute; inset: 0; pointer-events: all; cursor: crosshair; touch-action: none; }
    #box { position: absolute; border: 2px solid #0f0; background: rgba(0,255,0,0.1); display: none; box-sizing: border-box; }
    .panel { margin: 16px; padding: 16px; background: #222; border-radius: 8px; }
    .controls { display: flex; gap: 16px; align-items: flex-end; flex-wrap: wrap; margin-top: 12px; }
    .inputs { display: flex; gap: 8px; align-items: center; }
    .input-group { display: flex; flex-direction: column; }
    .input-group label { font-size: 0.8em; color: #aaa; margin-bottom: 2px; }
    input[type="number"] { width: 75px; background: #333; border: 1px solid #555; color: #eee; padding: 6px; border-radius: 4px; }
    .readonly-group input { background: #2a2a2a; color: #aaa; border-color: #444; }
    button { padding: 8px 16px; cursor: pointer; background: #444; color: #fff; border: 1px solid #666; border-radius: 4px; font-weight: bold; height: 36px; }
    button:hover { background: #555; }
    #status { margin-top: 8px; color: #aaa; font-style: italic; min-height: 1.2em; }
    h3 { margin: 0 0 8px 0; font-size: 1.1em; }
  </style>
</head>
<body>
  <div class="panel">
    <h3>Bounding Box Tool</h3>
    <div><strong>Instructions:</strong> Drag on image to create/adjust box. Normalized coordinates (0-1).</div>
    <div class="controls">
      <div class="inputs">
        <div class="input-group"><label>x1 (min_x)</label><input type="number" id="x1" step="0.001" min="0" max="1"></div>
        <div class="input-group"><label>y1 (min_y)</label><input type="number" id="y1" step="0.001" min="0" max="1"></div>
        <div class="input-group"><label>x2 (max_x)</label><input type="number" id="x2" step="0.001" min="0" max="1"></div>
        <div class="input-group"><label>y2 (max_y)</label><input type="number" id="y2" step="0.001" min="0" max="1"></div>
      </div>
      <div class="inputs readonly-group">
        <div class="input-group"><label>Width</label><input type="number" id="w" readonly></div>
        <div class="input-group"><label>Height</label><input type="number" id="h" readonly></div>
      </div>
      <button id="copy-btn">Copy [x1, y1, x2, y2]</button>
      <button id="clear-btn">Clear</button>
    </div>
    <div id="status"></div>
  </div>
  
  <div class="container" id="container">
    <img id="img" src="__IMG_NAME__" alt="target" draggable="false" />
    <div id="overlay">
      <div id="box"></div>
    </div>
  </div>

  <script>
    const box = document.getElementById('box');
    const overlay = document.getElementById('overlay');
    const img = document.getElementById('img');
    const inputs = {
      x1: document.getElementById('x1'),
      y1: document.getElementById('y1'),
      x2: document.getElementById('x2'),
      y2: document.getElementById('y2'),
    };
    const dims = {
      w: document.getElementById('w'),
      h: document.getElementById('h'),
    };
    const status = document.getElementById('status');
    
    let isDrawing = false;
    let startPos = null;

    // Prevent default drag behaviors
    img.addEventListener('dragstart', (e) => e.preventDefault());

    function getRect() {
      return overlay.getBoundingClientRect();
    }

    function updateBoxFromInputs() {
      const rect = getRect();
      if (rect.width === 0 || rect.height === 0) return;

      const x1 = parseFloat(inputs.x1.value) || 0;
      const y1 = parseFloat(inputs.y1.value) || 0;
      const x2 = parseFloat(inputs.x2.value) || 0;
      const y2 = parseFloat(inputs.y2.value) || 0;

      if (x1 === 0 && y1 === 0 && x2 === 0 && y2 === 0) {
         box.style.display = 'none';
         dims.w.value = '';
         dims.h.value = '';
         return;
      }
      
      const left = x1 * rect.width;
      const top = y1 * rect.height;
      const width = (x2 - x1) * rect.width;
      const height = (y2 - y1) * rect.height;
      
      dims.w.value = (x2 - x1).toFixed(4);
      dims.h.value = (y2 - y1).toFixed(4);

      box.style.display = 'block';
      box.style.left = left + 'px';
      box.style.top = top + 'px';
      box.style.width = width + 'px';
      box.style.height = height + 'px';
    }

    function setInputs(x1, y1, x2, y2) {
      inputs.x1.value = x1.toFixed(4);
      inputs.y1.value = y1.toFixed(4);
      inputs.x2.value = x2.toFixed(4);
      inputs.y2.value = y2.toFixed(4);
      
      dims.w.value = (x2 - x1).toFixed(4);
      dims.h.value = (y2 - y1).toFixed(4);
    }

    // Input listeners
    Object.values(inputs).forEach(inp => {
      inp.addEventListener('input', updateBoxFromInputs);
    });

    // Clear
    document.getElementById('clear-btn').addEventListener('click', () => {
      box.style.display = 'none';
      setInputs(0,0,0,0);
      dims.w.value = '';
      dims.h.value = '';
      status.textContent = 'Cleared.';
    });

    // Copy
    document.getElementById('copy-btn').addEventListener('click', () => {
      const vals = [
        parseFloat(inputs.x1.value) || 0,
        parseFloat(inputs.y1.value) || 0,
        parseFloat(inputs.x2.value) || 0,
        parseFloat(inputs.y2.value) || 0
      ];
      navigator.clipboard.writeText(JSON.stringify(vals));
      status.textContent = 'Copied to clipboard: ' + JSON.stringify(vals);
    });

    // Mouse events for drawing
    overlay.addEventListener('mousedown', (e) => {
      const rect = getRect();
      startPos = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      };
      isDrawing = true;
      box.style.display = 'block';
      
      e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDrawing) return;
      const rect = getRect();
      const currX = e.clientX - rect.left;
      const currY = e.clientY - rect.top;

      // Clamp to image bounds
      const x1_px = Math.max(0, Math.min(startPos.x, currX));
      const y1_px = Math.max(0, Math.min(startPos.y, currY));
      const x2_px = Math.min(rect.width, Math.max(startPos.x, currX));
      const y2_px = Math.min(rect.height, Math.max(startPos.y, currY));

      box.style.left = x1_px + 'px';
      box.style.top = y1_px + 'px';
      box.style.width = (x2_px - x1_px) + 'px';
      box.style.height = (y2_px - y1_px) + 'px';

      // Update inputs
      setInputs(
        x1_px / rect.width,
        y1_px / rect.height,
        x2_px / rect.width,
        y2_px / rect.height
      );
    });

    window.addEventListener('mouseup', () => {
      if (isDrawing) {
        isDrawing = false;
        status.textContent = 'Box set.';
      }
    });
    
    // Resize observer
    new ResizeObserver(() => {
        // Debounce or just run?
        requestAnimationFrame(updateBoxFromInputs);
    }).observe(img);
  </script>
</body>
</html>
"""


def build_site(image_path: Path) -> str:
    tmpdir = tempfile.mkdtemp(prefix="box_selector_")
    target_image = Path(tmpdir) / "image.png"
    shutil.copy(image_path, target_image)

    html = HTML_TEMPLATE.replace("__IMG_NAME__", target_image.name)
    (Path(tmpdir) / "index.html").write_text(html, encoding="utf-8")
    return tmpdir


def serve(directory: str, port: int) -> None:
    handler = http.server.SimpleHTTPRequestHandler
    os.chdir(directory)
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at http://localhost:{port}")
        print("Press Ctrl+C to stop.")
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Host a simple web page to draw a bounding box on an image."
    )
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on.")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    # Validate image loadable
    Image.open(image_path).verify()

    site_dir = build_site(image_path)
    try:
        serve(site_dir, args.port)
    finally:
        shutil.rmtree(site_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
