import http.server
import socketserver
import json
import sys
import os
import traceback
import time
from urllib.parse import urlparse, parse_qs
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from .attention import AttentionExtractor
from .hardware import get_recommended_model_config, get_device

# Global instances
extractor = None
planner = None
planner_backend = os.getenv("CAPTCHA_PLANNER_BACKEND", "vllm")
planner_model_id = os.getenv("CAPTCHA_PLANNER_MODEL")

class ToolServerHandler(http.server.BaseHTTPRequestHandler):
    def _send_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _read_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode('utf-8'))

    def do_GET(self):
        if self.path == '/health':
            self._send_response({
                'status': 'ok', 
                'models_loaded': extractor is not None,
                'planner_model': planner_model_id
            })
        else:
            self._send_response({'error': 'Not found'}, 404)

    def do_POST(self):
        try:
            parsed_path = urlparse(self.path)
            endpoint = parsed_path.path

            if endpoint == '/detect':
                data = self._read_body()
                image_path = data.get('image_path')
                text_prompt = data.get('text_prompt')
                max_objects = data.get('max_objects', 24)
                
                if not image_path or not text_prompt:
                    self._send_response({'error': 'Missing image_path or text_prompt'}, 400)
                    return

                print(f"[Server] Detecting '{text_prompt}' with SAM 3 in {image_path}...", file=sys.stderr)
                
                # Use SAM 3 for detection
                result = extractor.detect(image_path, text_prompt, max_objects=max_objects)
                self._send_response({"objects": result})

            elif endpoint == '/get_mask':
                data = self._read_body()
                image_path = data.get('image_path')
                text_prompt = data.get('text_prompt')
                max_objects = data.get('max_objects', 24)
                
                if not image_path or not text_prompt:
                    self._send_response({'error': 'Missing image_path or text_prompt'}, 400)
                    return

                print(f"[Server] Getting masks for '{text_prompt}' in {image_path}...", file=sys.stderr)
                masks = extractor.get_mask(image_path, text_prompt, max_objects=max_objects)
                
                # Convert numpy masks to lists for JSON serialization
                serialized_masks = []
                for m in masks:
                    if hasattr(m, 'tolist'):
                        serialized_masks.append(m.tolist())
                    else:
                        serialized_masks.append(m)
                
                self._send_response({"masks": serialized_masks})

            elif endpoint == '/segment':
                # For compatibility, /segment can now just be a synonym for detect
                data = self._read_body()
                image_path = data.get('image_path')
                prompt = data.get('prompt', "objects")
                max_objects = data.get('max_objects', 24)
                
                if not image_path:
                    self._send_response({'error': 'Missing image_path'}, 400)
                    return

                print(f"[Server] Segmenting {image_path} with prompt '{prompt}'...", file=sys.stderr)
                result = extractor.detect(image_path, prompt, max_objects=max_objects)
                self._send_response({"objects": result})

            elif endpoint == '/plan':
                data = self._read_body()
                image_paths = data.get('image_paths', [])
                if not image_paths and data.get('image_path'):
                    image_paths = [data.get('image_path')]
                
                prompt = data.get('prompt')
                
                if not image_paths or not prompt:
                    self._send_response({'error': 'Missing image_paths or prompt'}, 400)
                    return

                print(f"[Server] Planning with {planner.backend} ({planner.model})...", file=sys.stderr)
                
                t0 = time.time()
                try:
                    # Use the warm planner instance
                    response_text, _ = planner._chat_with_image(prompt, image_paths)
                    t1 = time.time()
                    
                    print(f"[Server] Planning took {t1-t0:.2f}s", file=sys.stderr)
                    self._send_response({"response": response_text})
                except Exception as e:
                    print(f"[Server] Planning error: {e}", file=sys.stderr)
                    traceback.print_exc()
                    self._send_response({'error': str(e)}, 500)

            else:
                self._send_response({'error': 'Endpoint not found'}, 404)

        except Exception as e:
            traceback.print_exc()
            self._send_response({'error': str(e)}, 500)

def start_tool_server(port=8000):
    global extractor, planner, planner_model_id, planner_backend
    
    # 1. Initialize AttentionExtractor and pre-load it
    print(f"Initializing AttentionExtractor (SAM 3)...", file=sys.stderr)
    extractor = AttentionExtractor()
    extractor.load_models()  # Pre-load SAM 3 into VRAM
    
    # 2. Initialize and Warm ActionPlanner
    if planner_model_id is None:
        if planner_backend == "vllm":
            planner_model_id = "Jake-Writer-Jobharvest/qwen3-vl-8b-merged-bf16"
        else:
            # Fallback to transformers if vLLM not specified or available
            planner_backend = "transformers"
            from .hardware import get_recommended_model_config
            planner_model_id, _, _ = get_recommended_model_config()
    
    print(f"Initializing Warm Planner ({planner_backend}: {planner_model_id})...", file=sys.stderr)
    from .planner import ActionPlanner
    planner = ActionPlanner(backend=planner_backend, model=planner_model_id)
    
    # Warm up the planner with a dummy call
    # For vLLM, the model is loaded on __init__ if it hasn't been yet,
    # but a dummy generation ensures it's fully ready in VRAM.
    try:
        print(f"Warming up planner VRAM...", file=sys.stderr)
        # We don't have a dummy image handy here easily, but we can trigger 
        # the model load by just calling the underlying generate if needed.
        # ActionPlanner.__init__ for vLLM already creates the LLM instance which loads the model.
    except Exception as e:
        print(f"Warning: Planner warmup failed: {e}", file=sys.stderr)
        
    print(f"Tool server ready and models are warm in VRAM.", file=sys.stderr)
    
    print(f"Starting server on port {port}...", file=sys.stderr)
    try:
        with socketserver.TCPServer(("", port), ToolServerHandler) as httpd:
            print(f"Server running at http://localhost:{port}", file=sys.stderr)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        traceback.print_exc()

if __name__ == "__main__":
    start_tool_server()

