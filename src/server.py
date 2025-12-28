import http.server
import socketserver
import json
import sys
import os
import traceback
from urllib.parse import urlparse, parse_qs
from src.attention import AttentionExtractor

# Global extractor instance
extractor = None

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
            self._send_response({'status': 'ok', 'models_loaded': True})
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

                print(f"[Server] Detecting '{text_prompt}' in {image_path}...", file=sys.stderr)
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

            elif endpoint == '/shutdown':
                 self._send_response({'status': 'shutting down'})
                 sys.exit(0)

            else:
                self._send_response({'error': 'Endpoint not found'}, 404)

        except Exception as e:
            traceback.print_exc()
            self._send_response({'error': str(e)}, 500)

def start_tool_server(port=8000):
    global extractor
    print(f"Initializing AttentionExtractor...", file=sys.stderr)
    extractor = AttentionExtractor()
    print(f"Loading models eagerly...", file=sys.stderr)
    extractor.load_models()
    
    print(f"Starting server on port {port}...", file=sys.stderr)
    with socketserver.TCPServer(("", port), ToolServerHandler) as httpd:
        print(f"Server running at http://localhost:{port}", file=sys.stderr)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.", file=sys.stderr)

if __name__ == "__main__":
    start_tool_server()

