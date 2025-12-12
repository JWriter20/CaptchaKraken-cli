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
                box_threshold = data.get('box_threshold', 0.35)
                text_threshold = data.get('text_threshold', 0.25)
                
                if not image_path or not text_prompt:
                    self._send_response({'error': 'Missing image_path or text_prompt'}, 400)
                    return

                print(f"[Server] Detecting '{text_prompt}' in {image_path}...", file=sys.stderr)
                result = extractor.detect_with_grounding_dino(
                    image_path, text_prompt, box_threshold, text_threshold
                )
                self._send_response(result)

            elif endpoint == '/segment':
                data = self._read_body()
                image_path = data.get('image_path')
                points = data.get('points') # List[List[float]]
                boxes = data.get('boxes')   # List[List[float]]
                
                if not image_path:
                    self._send_response({'error': 'Missing image_path'}, 400)
                    return

                print(f"[Server] Segmenting {image_path}...", file=sys.stderr)
                # Note: masks are numpy arrays, need to convert to list/rle for JSON
                # specific to server usage, we might just return counts or something
                # But for now let's convert masks to list of lists (boolean)
                
                result = extractor.segment_with_sam2(
                    image_path, input_points=points, input_boxes=boxes
                )
                
                # Convert numpy masks to lists for JSON serialization
                masks = result.get('masks', [])
                serialized_masks = []
                for m in masks:
                    if hasattr(m, 'tolist'):
                        serialized_masks.append(m.tolist())
                    else:
                        serialized_masks.append(m)
                
                result['masks'] = serialized_masks
                
                if 'points' in result and hasattr(result['points'], 'tolist'):
                    result['points'] = result['points'].tolist()

                self._send_response(result)

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

