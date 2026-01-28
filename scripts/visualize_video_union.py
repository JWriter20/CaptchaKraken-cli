import cv2
import os
import sys
import tempfile
from PIL import Image
import numpy as np

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.attention import AttentionExtractor
from src.overlay import add_overlays_to_image

class SimpleTracker:
    """Simple centroid tracker to maintain object ID consistency across frames."""
    def __init__(self, dist_threshold=0.15):
        self.prev_centroids = {} # id -> (x, y)
        self.next_id = 1
        self.dist_threshold = dist_threshold

    def update(self, current_detections):
        tracked = []
        new_centroids = {}
        
        # Sort current detections by area descending to match larger objects first
        current_detections.sort(key=lambda d: (d['x_max']-d['x_min'])*(d['y_max']-d['y_min']), reverse=True)
        
        assigned_ids = set()
        
        for det in current_detections:
            x1, y1, x2, y2 = det['x_min'], det['y_min'], det['x_max'], det['y_max']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            best_id = None
            min_dist = float('inf')
            
            # Find closest previous centroid that hasn't been reassigned
            for pid, (px, py) in self.prev_centroids.items():
                if pid in assigned_ids: continue
                
                dist = ((cx - px)**2 + (cy - py)**2)**0.5
                if dist < min_dist and dist < self.dist_threshold:
                    min_dist = dist
                    best_id = pid
            
            if best_id is not None:
                assigned_id = best_id
                assigned_ids.add(assigned_id)
            else:
                assigned_id = self.next_id
                self.next_id += 1
            
            new_centroids[assigned_id] = (cx, cy)
            det_with_id = det.copy()
            det_with_id['id'] = assigned_id
            tracked.append(det_with_id)
            
        self.prev_centroids = new_centroids
        return tracked

def process_video(input_path, output_path, prompt, max_frames=30):
    print(f"Initializing AttentionExtractor and Tracker...")
    extractor = AttentionExtractor()
    tracker = SimpleTracker()
    
    print(f"Opening video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    actual_max = min(total_frames, max_frames) if total_frames > 0 else max_frames
    
    try:
        while cap.isOpened() and frame_idx < actual_max:
            ret, frame = cap.read()
            if not ret:
                break
                
            print(f"Processing frame {frame_idx + 1}/{actual_max}...")
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                temp_frame_path = tf.name
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                # 1. Detect (includes Union-based clustering in src/attention.py)
                detections = extractor.detect(temp_frame_path, prompt)
                
                # 2. Track
                tracked_detections = tracker.update(detections)
                
                # 3. Prepare Overlays
                overlays = []
                for det in tracked_detections:
                    overlays.append({
                        "bbox": [det['x_min'], det['y_min'], det['x_max'], det['y_max']],
                        "number": det['id'],
                        "color": "#FF0000",
                        "box_style": "thin"
                    })
                
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf_labeled:
                    labeled_frame_path = tf_labeled.name
                
                add_overlays_to_image(temp_frame_path, overlays, output_path=labeled_frame_path, label_position="top-right")
                
                labeled_frame = cv2.imread(labeled_frame_path)
                if labeled_frame is not None:
                    out.write(labeled_frame)
                else:
                    out.write(frame)
                    
            finally:
                if os.path.exists(temp_frame_path): os.remove(temp_frame_path)
                if 'labeled_frame_path' in locals() and os.path.exists(labeled_frame_path): 
                    os.remove(labeled_frame_path)
            
            frame_idx += 1
            
    finally:
        cap.release()
        out.release()
        
    print(f"\nFinished processing {frame_idx} frames.")
    print(f"Output saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize SAM3 detections on video with object tracking.")
    parser.add_argument("--input", default=os.path.join(PROJECT_ROOT, "captchaimages", "hcaptcha_1766539373078.webm"), help="Input video path")
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "video_verification.webm"), help="Output video path")
    parser.add_argument("--prompt", default="pink object", help="SAM3 prompt")
    parser.add_argument("--frames", type=int, default=30, help="Max frames to process")
    
    args = parser.parse_args()
    process_video(args.input, args.output, args.prompt, max_frames=args.frames)

