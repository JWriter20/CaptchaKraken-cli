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
planner_model = None
planner_processor = None
planner_model_id = None

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

                print(f"[Server] Planning with Ollama (test-qwen3-vl:latest)...", file=sys.stderr)
                
                import ollama
                
                t0 = time.time()
                try:
                    # Use generate() for maximum speed, remove format='json' to avoid slow constrained decoding
                    # We will parse the JSON ourselves.
                    response = ollama.generate(
                        model="test-qwen3-vl:latest",
                        prompt=prompt,
                        images=image_paths, # Path-based is usually faster
                        options={
                            "temperature": 0.0,
                            "num_predict": 128
                        }
                    )
                    output_text = response['response']
                    t1 = time.time()
                    
                    eval_count = response.get('eval_count', 0)
                    eval_duration = response.get('eval_duration', 1) / 1e9
                    prompt_eval_count = response.get('prompt_eval_count', 0)
                    prompt_eval_duration = response.get('prompt_eval_duration', 1) / 1e9
                    
                    print(f"[Server] Ollama planning took {t1-t0:.2f}s (Prompt: {prompt_eval_count} tokens in {prompt_eval_duration:.2f}s, Gen: {eval_count} tokens in {eval_duration:.2f}s, Rate: {eval_count/max(0.001, eval_duration):.1f} tok/s)", file=sys.stderr)
                    self._send_response({"response": output_text})
                except Exception as e:
                    print(f"[Server] Ollama error: {e}", file=sys.stderr)
                    self._send_response({'error': str(e)}, 500)

            else:
                self._send_response({'error': 'Endpoint not found'}, 404)

        except Exception as e:
            traceback.print_exc()
            self._send_response({'error': str(e)}, 500)

def start_tool_server(port=8000):
    global extractor, planner_model_id
    
    # 1. Initialize AttentionExtractor and pre-load it
    print(f"Initializing AttentionExtractor...", file=sys.stderr)
    extractor = AttentionExtractor()
    extractor.load_models()  # Pre-load SAM 3 into VRAM
    
    # 2. Set Planner Model ID (Ollama handled in /plan endpoint)
    planner_model_id = "test-qwen3-vl:latest"
    print(f"Warming up Ollama ({planner_model_id})...", file=sys.stderr)
    import ollama
    try:
        ollama.generate(model=planner_model_id, prompt="hi")
    except:
        pass
    print(f"Tool server ready.", file=sys.stderr)
    
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

