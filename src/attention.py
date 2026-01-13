"""
AttentionExtractor - Wrapper around SAM 3 for detection and masking.

Exposes:
- detect(): Find all instances of an object class using SAM 3 (returns bounding boxes)
- get_mask(): Get segmentation masks for an object class using SAM 3
- clusterDetections(): Merge overlapping/close detection boxes
"""

from __future__ import annotations

import os
import sys
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from .image_processor import ImageProcessor
except (ImportError, ValueError):
    try:
        from image_processor import ImageProcessor
    except ImportError:
        ImageProcessor = None

if TYPE_CHECKING:
    pass


def clusterDetections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    distance_threshold: Optional[float] = 0.04,
) -> List[Dict[str, Any]]:
    """
    Cluster and merge overlapping or very close detection bounding boxes.

    Can use either IoU-based clustering (for overlapping boxes) or
    distance-based clustering (for nearby but non-overlapping boxes).

    Args:
        detections: List of bounding boxes with keys: x_min, y_min, x_max, y_max
        iou_threshold: IoU threshold for merging (default: 0.5)
        distance_threshold: If set, use center distance instead of IoU.
                          Boxes with centers closer than this are merged.
                          (Default: 0.04 for normalized coordinates)

    Returns:
        List of merged bounding boxes in same format
    """
    if not detections:
        return []

    def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
        """Calculate Intersection over Union between two boxes."""
        # Calculate intersection area
        x_min_inter = max(box1["x_min"], box2["x_min"])
        y_min_inter = max(box1["y_min"], box2["y_min"])
        x_max_inter = min(box1["x_max"], box2["x_max"])
        y_max_inter = min(box1["y_max"], box2["y_max"])

        if x_max_inter < x_min_inter or y_max_inter < y_min_inter:
            return 0.0

        intersection = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)

        # Calculate union area
        area1 = (box1["x_max"] - box1["x_min"]) * (box1["y_max"] - box1["y_min"])
        area2 = (box2["x_max"] - box2["x_min"]) * (box2["y_max"] - box2["y_min"])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def calculate_center_distance(box1: Dict[str, float], box2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between box centers."""
        center1_x = (box1["x_min"] + box1["x_max"]) / 2
        center1_y = (box1["y_min"] + box1["y_max"]) / 2
        center2_x = (box2["x_min"] + box2["x_max"]) / 2
        center2_y = (box2["y_min"] + box2["y_max"]) / 2

        return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5

    def merge_cluster_boxes(boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Encompass all boxes in the cluster (Union) and average scores."""
        n = len(boxes)
        result = {
            "x_min": min(b["x_min"] for b in boxes),
            "y_min": min(b["y_min"] for b in boxes),
            "x_max": max(b["x_max"] for b in boxes),
            "y_max": max(b["y_max"] for b in boxes),
        }
        if "score" in boxes[0]:
            result["score"] = sum(b["score"] for b in boxes) / n
        return result

    # Greedy clustering: merge boxes based on IoU or distance
    clusters = []
    used = [False] * len(detections)

    # Use BOTH IoU and distance for robustness
    def should_merge(box1: Dict[str, Any], box2: Dict[str, Any]) -> bool:
        iou = calculate_iou(box1, box2)
        dist = calculate_center_distance(box1, box2)
        
        # Merge if they overlap significantly OR if centers are close
        return iou > iou_threshold or (distance_threshold is not None and dist < distance_threshold)

    for i in range(len(detections)):
        if used[i]:
            continue

        # Start a new cluster
        cluster = [detections[i]]
        used[i] = True

        # Find all boxes that should merge with any box in the cluster
        changed = True
        while changed:
            changed = False
            for j in range(len(detections)):
                if used[j]:
                    continue

                # Check if this box should merge with any box in the cluster
                for cluster_box in cluster:
                    if should_merge(detections[j], cluster_box):
                        cluster.append(detections[j])
                        used[j] = True
                        changed = True
                        break

        clusters.append(cluster)

    # Use union (encompassing box) instead of average to avoid cutting off parts
    merged = [merge_cluster_boxes(cluster) for cluster in clusters]

    return merged


def default_predicate(detection: Dict[str, Any]) -> Optional[float]:
    """
    Refined filtering and weighting for detections.
    - Ignores boxes with width or height > 30% of image (requested change).
    - Filters out very tiny detections (< 0.5% of image dimension).
    - Penalizes very rectangular shapes (height > 2x width or vice versa).
    - Penalizes boxes > 20% dimension and very small boxes (< 5% dimension).
    - Boosts score for square-like boxes.
    """
    if "bbox" in detection:
        x1, y1, x2, y2 = detection["bbox"]
    else:
        x1, y1, x2, y2 = detection["x_min"], detection["y_min"], detection["x_max"], detection["y_max"]

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    # Start with a healthy baseline
    score = detection.get("score", 0.6) 

    # 1. Hard Filters
    # Too big (> 30% in either dimension - user requested)
    if w > 0.30 or h > 0.30:
        return None
    # Too tiny (< 0.5% in either dimension - original "good" logic)
    if w < 0.005 or h < 0.005:
        return None

    # Too tall/wide (> 3x in either dimension - original "good" logic)
    if w > h * 3 or h > w * 3:
        return None

    # 2. Aspect Ratio Penalties
    if w > 0 and h > 0:
        aspect_ratio = w / h
        if aspect_ratio > 2.0 or aspect_ratio < 0.5:
            score -= 0.1 # Penalty for very thin/flat shapes
        
        # Squareness boost (up to 0.1)
        squareness = min(w, h) / max(w, h)
        score += squareness * 0.1

    # 3. Size-based Penalties
    # Large but not filtered (20-30%)
    if w > 0.20 or h > 0.20:
        score -= 0.1
    # Tiny but not filtered (0.5-2.5%)
    if w < 0.025 or h < 0.025:
        score -= 0.10

    if score < 0.1:
        return None

    return score


class SimpleTracker:
    """Simple centroid tracker to maintain object ID consistency across frames."""

    def __init__(self, dist_threshold: float = 0.15):
        self.prev_centroids: Dict[int, Tuple[float, float]] = {}  # id -> (x, y)
        self.next_id = 1
        self.dist_threshold = dist_threshold

    def update(self, current_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tracked = []
        new_centroids = {}

        # Sort current detections by area descending to match larger objects first
        current_detections.sort(
            key=lambda d: (d["x_max"] - d["x_min"]) * (d["y_max"] - d["y_min"]), reverse=True
        )

        assigned_ids = set()

        for det in current_detections:
            x1, y1, x2, y2 = det["x_min"], det["y_min"], det["x_max"], det["y_max"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            best_id = None
            min_dist = float("inf")

            # Find closest previous centroid that hasn't been reassigned
            for pid, (px, py) in self.prev_centroids.items():
                if pid in assigned_ids:
                    continue

                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
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
            det_with_id["id"] = assigned_id
            tracked.append(det_with_id)

        self.prev_centroids = new_centroids
        return tracked


class AttentionExtractor:
    """Wrapper around SAM 3 for bounding box detection."""

    def __init__(
        self,
        device: Optional[str] = None,
        **kwargs,
    ):
        # Auto-detect device
        if device is None:
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
                if hasattr(torch.version, "hip") and torch.version.hip:
                    print(f"[AttentionExtractor] Using AMD GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._sam3_model = None
        self._sam3_processor = None
        self._sam3_video_model = None
        self._sam3_video_processor = None
        self._sam3_auto_generator = None

    def unload_models(self, models: Optional[List[str]] = None):
        """
        Unload specific models or all models to free memory.
        
        Args:
            models: List of model names to unload ("sam3").
                   If None, unloads all.
        """
        import gc
        import torch

        all_models = {
            "sam3": ["_sam3_model", "_sam3_processor", "_sam3_video_model", "_sam3_video_processor", "_sam3_auto_generator"],
        }
        
        targets = models if models else all_models.keys()
        
        for name in targets:
            if name in all_models:
                print(f"[AttentionExtractor] Unloading {name}...", file=sys.stderr)
                for attr in all_models[name]:
                    if hasattr(self, attr):
                        setattr(self, attr, None)
                        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
             if hasattr(torch.mps, "empty_cache"):
                 torch.mps.empty_cache()

    def load_models(self):
        """Eagerly load all models (SAM 3)."""
        self._load_sam3()

    def _load_sam3(self):
        """Lazy load SAM 3 model and processor."""
        if self._sam3_model is not None:
            return

        t0 = time.time()
        import torch
        from transformers import Sam3Model, Sam3Processor, Sam3VideoModel, Sam3VideoProcessor

        model_id = "facebook/sam3"
        print(f"[AttentionExtractor] Loading {model_id}...", file=sys.stderr)

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        # Load image model/processor
        self._sam3_model = Sam3Model.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(self.device)
        self._sam3_processor = Sam3Processor.from_pretrained(model_id, trust_remote_code=True)

        # Load video model/processor
        self._sam3_video_model = Sam3VideoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(self.device)
        self._sam3_video_processor = Sam3VideoProcessor.from_pretrained(model_id, trust_remote_code=True)

        t1 = time.time()
        print(f"[AttentionExtractor] Loaded SAM 3 models in {t1 - t0:.2f}s", file=sys.stderr)

    def _load_sam3_auto(self):
        """Lazy load SAM 3 automatic mask generator pipeline."""
        if self._sam3_auto_generator is not None:
            return
        
        from transformers import pipeline
        print(f"[AttentionExtractor] Loading SAM 3 mask-generation pipeline...", file=sys.stderr)
        self._sam3_auto_generator = pipeline(
            "mask-generation", 
            model="facebook/sam3", 
            device=self.device, 
            trust_remote_code=True
        )

    def detect(
        self,
        media_path: str,
        object_description: str,
        max_objects: int = 24,
        max_frames: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Detect all instances of object class using SAM 3.
        Delegates to the implementation in tool_calls.detect.
        """
        try:
            from .tool_calls.detect import detect as run_detect
        except (ImportError, ValueError):
            from tool_calls.detect import detect as run_detect
            
        return run_detect(
            self, 
            media_path, 
            object_description, 
            max_objects=max_objects, 
            max_frames=max_frames
        )

    def detect_all(
        self,
        media_path: str,
        max_objects: int = 8,
        min_score_threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Detect all significant items in the image using parallelized multi-prompt
        and automatic mask generation.
        Filters out detections in the top and bottom 15% of the image.
        """
        t0 = time.time()
        all_detections = []

        # Simplified but effective prompt list
        prompts = ["item", "object", "animal", "symbol"]

        def run_prompt_detect(prompt):
            try:
                # Use a very low threshold for raw detections to catch everything
                return self.detect(media_path, prompt, max_objects=24)
            except Exception as e:
                print(f"[AttentionExtractor] Prompt '{prompt}' failed: {e}", file=sys.stderr)
                return []

        def run_auto_detect():
            try:
                # Unload prompt models to free memory for the automatic generator
                self.unload_models(["sam3"])
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self._load_sam3_auto()
                image = Image.open(media_path).convert("RGB")
                width, height = image.size
                
                # Higher point density to find missing objects
                auto_results = self._sam3_auto_generator(image, points_per_side=32)
                auto_masks = auto_results.get("masks", [])
                
                auto_detections = []
                for mask in auto_masks:
                    if hasattr(mask, "cpu"): mask = mask.cpu().numpy()
                    if len(mask.shape) == 3: mask = mask[0]
                    
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if not np.any(rows) or not np.any(cols): continue
                    
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    
                    auto_detections.append({
                        "x_min": float(cmin) / width,
                        "y_min": float(rmin) / height,
                        "x_max": float(cmax) / width,
                        "y_max": float(rmax) / height,
                        "score": 0.6,
                    })
                return auto_detections
            except Exception as e:
                print(f"[AttentionExtractor] Automatic detection failed: {e}", file=sys.stderr)
                return []

        # 1. Run prompt detections
        for prompt in prompts:
            all_detections.extend(run_prompt_detect(prompt))

        # 2. Run automatic detection (unloads prompts first)
        all_detections.extend(run_auto_detect())

        # Merge overlapping detections with stricter thresholds
        # distance_threshold=0.01 is very strict to keep nearby but distinct objects separate
        merged = clusterDetections(all_detections, iou_threshold=0.5, distance_threshold=0.01)
        
        final_results = []
        for det in merged:
            # 1. Coordinate Filter: 15% top/bottom buffer
            y_center = (det["y_min"] + det["y_max"]) / 2
            if y_center < 0.15 or y_center > 0.85:
                continue

            # 2. Refined Predicate (Size, Aspect Ratio, Score Boosts)
            score = default_predicate(det)
            if score is not None and score >= min_score_threshold:
                det["score"] = score
                if "bbox" not in det:
                    det["bbox"] = [det["x_min"], det["y_min"], det["x_max"], det["y_max"]]
                final_results.append(det)
        
        # Sort by score descending and return top N
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        t1 = time.time()
        print(f"[AttentionExtractor] detect_all found {len(final_results)} objects in {t1-t0:.2f}s", file=sys.stderr)
        
        return final_results[:max_objects]

    def get_mask(
        self,
        media_path: str,
        object_description: Optional[str] = None,
        bboxes: Optional[List[List[float]]] = None,
        points: Optional[List[List[float]]] = None,
        max_objects: int = 24,
    ) -> List[np.ndarray]:
        """
        Get segmentation masks for an object class using SAM 3.
        Supports both images and videos.
        
        Args:
            media_path: Path to media
            object_description: Optional text prompt
            bboxes: Optional list of bounding boxes in [x1, y1, x2, y2] normalized
            points: Optional list of points in [x, y] normalized
            max_objects: Max masks to return
        """
        import torch
        # Check for tool server
        tool_server_url = os.getenv("CAPTCHA_TOOL_SERVER")
        if tool_server_url:
            import requests
            try:
                abs_path = os.path.abspath(media_path)
                resp = requests.post(
                    f"{tool_server_url}/get_mask",
                    json={
                        "image_path": abs_path, 
                        "text_prompt": object_description, 
                        "bboxes": bboxes,
                        "points": points,
                        "max_objects": max_objects
                    },
                    timeout=300
                )
                resp.raise_for_status()
                masks_data = resp.json().get("masks", [])
                return [np.array(m) for m in masks_data]
            except Exception as e:
                print(f"[AttentionExtractor] Tool server get_mask failed: {e}. Falling back to local.", file=sys.stderr)

        self._load_sam3()
        
        is_video = any(media_path.lower().endswith(ext) for ext in [".mp4", ".webm", ".gif", ".avi"])
        if is_video:
            from transformers.video_utils import load_video
            video_frames, _ = load_video(media_path)
            image = video_frames[0]
        else:
            image = Image.open(media_path).convert("RGB")
        
        inputs_kwargs = {"images": image, "return_tensors": "pt"}
        if object_description:
            inputs_kwargs["text"] = object_description
        
        width, height = image.size
        if bboxes:
            # Convert normalized [x1, y1, x2, y2] to pixel coordinates
            scaled_bboxes = [[b[0] * width, b[1] * height, b[2] * width, b[3] * height] for b in bboxes]
            # Convert [N, 4] to [1, N, 4] for transformers
            inputs_kwargs["input_boxes"] = [scaled_bboxes]
        if points:
            # SAM 3 might not support points directly in some versions, or uses boxes instead.
            # Convert points to small normalized boxes [cx, cy, w, h] as SAM 3 forward docs suggest
            # or [x1, y1, x2, y2] if the processor expects that.
            # Looking at Sam3Processor docs, it takes 'input_boxes'.
            # We'll use very small boxes (2% of image size) centered at the points.
            box_size = 0.02 
            small_boxes = []
            for p in points:
                x, y = p[0], p[1]
                # [x1, y1, x2, y2] normalized
                x1 = max(0, x - box_size/2)
                y1 = max(0, y - box_size/2)
                x2 = min(1.0, x + box_size/2)
                y2 = min(1.0, y + box_size/2)
                small_boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
            
            inputs_kwargs["input_boxes"] = [small_boxes]
            inputs_kwargs["input_boxes_labels"] = [[1] * len(points)]
            
        inputs = self._sam3_processor(**inputs_kwargs).to(self.device)
        
        # Ensure floating point inputs match model dtype (e.g., BFloat16)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                inputs[k] = v.to(self._sam3_model.dtype)
        with torch.no_grad():
            outputs = self._sam3_model(**inputs)

        # Post-process results
        results = self._sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.1,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        masks = results["masks"] # [num_objs, H, W]
        scores = results["scores"]
        
        # Sort by score and take top N
        top_indices = scores.argsort(descending=True)[:max_objects]
        
        print(f"[AttentionExtractor] get_mask() found {len(scores)} candidate masks. Top scores: {scores[top_indices].tolist()}", file=sys.stderr)
        
        final_masks = []
        for idx in top_indices:
            mask = masks[idx]
            if hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            final_masks.append(mask)
            
        return final_masks

    def segment_by_color(
        self,
        image_path: str,
        k: int = 8,
        min_area_ratio: float = 0.005,
        merge_components: bool = False,
    ) -> Dict[str, Any]:
        """
        Segment image using K-Means color clustering.
        Useful for distinct simple shapes where SAM 2 might over-segment texture.
        
        Args:
            image_path: Path to image
            k: Number of color clusters
            min_area_ratio: Minimum area fraction to keep a mask
            merge_components: If True, keep all regions of same color as one mask.
                            If False, split disjoint regions into separate masks.
        """
        import cv2
        
        t0 = time.time()
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"[AttentionExtractor] Could not load {image_path}", file=sys.stderr)
            return {"masks": []}
            
        # Use Mean Shift Filtering for better segmentation
        img_blur = cv2.pyrMeanShiftFiltering(img, sp=15, sr=30)
        
        # Convert to LAB for better color perceptual distance
        img_lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
        
        # Reshape to list of pixels
        pixels = img_lab.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # Define criteria = ( type, max_iter, epsilon )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # K-Means
        # k clusters
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers back to RGB for consistency (output uses RGB)
        centers_lab = np.uint8(centers).reshape(1, k, 3)
        centers_rgb = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB).reshape(k, 3)
        
        # Convert back to image shape
        height, width = img.shape[:2]
        labels_img = labels.reshape((height, width))
        
        masks = []
        mask_colors = []
        total_area = height * width
        
        for i in range(k):
            # Create binary mask for this cluster
            # labels_img is (H, W), mask is boolean
            cluster_mask = (labels_img == i).astype(np.uint8) * 255
            color = centers_rgb[i].tolist() # [R, G, B]
            
            if merge_components:
                # Still filter tiny specks even if merging
                if np.sum(cluster_mask > 0) < (total_area * min_area_ratio):
                    continue
                masks.append(cluster_mask > 0)
                mask_colors.append(color)
            else:
                # Split connected components
                num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(cluster_mask, connectivity=8)
                
                # label 0 is background (which is 0 in cluster_mask)
                # We want the foreground components (labels 1..N)
                
                for j in range(1, num_labels):
                    area = stats[j, cv2.CC_STAT_AREA]
                    # INCREASED filtering: skip small noise
                    # min_area_ratio default was 0.005 (0.5%).
                    # For "larger continuous segments", let's use the passed param, 
                    # but caller should likely increase it.
                    if area < (total_area * min_area_ratio):
                        continue
                        
                    component_mask = (labels_cc == j)
                    masks.append(component_mask)
                    mask_colors.append(color)
        
        t1 = time.time()
        print(f"[AttentionExtractor] Color segmentation (k={k}) took {t1 - t0:.2f}s, found {len(masks)} masks", file=sys.stderr)
        
        return {"masks": masks, "colors": mask_colors}

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def visualize_attention(
        self,
        image_path: str,
        target_description: str,
        output_path: str = "attention_heatmap.png",
        show_peak: bool = True,
    ) -> str:
        """Generate visualization showing detected bounding box."""
        import matplotlib.pyplot as plt

        detections = self.detect(image_path, target_description, max_objects=1)
        if detections:
            obj = detections[0]
            # Handle both formats: dict with keys or dict with 'bbox' list
            if "bbox" in obj:
                x1_pct, y1_pct, x2_pct, y2_pct = obj["bbox"]
            else:
                x1_pct, y1_pct, x2_pct, y2_pct = obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]
        else:
            x1_pct, y1_pct, x2_pct, y2_pct = 0.45, 0.45, 0.55, 0.55

        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)
        img_height, img_width = img_array.shape[:2]

        # Convert percentage to pixels for visualization
        px1, py1 = int(x1_pct * img_width), int(y1_pct * img_height)
        px2, py2 = int(x2_pct * img_width), int(y2_pct * img_height)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.imshow(img_array)

        if show_peak:
            # Draw bounding box
            rect = plt.Rectangle((px1, py1), px2 - px1, py2 - py1, 
                                 linewidth=3, edgecolor='lime', facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Label
            ax.text(px1, py1 - 5, f'Target: "{target_description}"', 
                    color='lime', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))

        ax.set_title(f'Target Detection: "{target_description}"\nBox: [{x1_pct:.2%}, {y1_pct:.2%}, {x2_pct:.2%}, {y2_pct:.2%}]')
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[AttentionExtractor] Saved visualization: {output_path}")
        return output_path

    def visualize_detections(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        output_path: str = "detections.png",
        elements: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Visualize detected objects with bounding boxes."""
        try:
            from .overlay import add_overlays_to_image
        except (ImportError, ValueError):
            from overlay import add_overlays_to_image

        # Load image to get dimensions for coordinate conversion
        with Image.open(image_path) as image:
            img_width, img_height = image.size

        boxes = []
        # Use darker colors for better contrast with white text
        colors = [
            "#C0392B",  # Dark Red
            "#16A085",  # Dark Teal
            "#2980B9",  # Dark Blue
            "#8E44AD",  # Dark Purple
            "#2C3E50",  # Dark Blue-Grey
            "#D35400",  # Pumpkin/Dark Orange
            "#27AE60",  # Dark Green
            "#7F8C8D",  # Grey
        ]

        for i, det in enumerate(detections):
            # Handle both formats: dict with keys or dict with 'bbox' list
            if "bbox" in det:
                bbox = det["bbox"]
                x1_pct, y1_pct, x2_pct, y2_pct = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                x1_pct, y1_pct, x2_pct, y2_pct = det["x_min"], det["y_min"], det["x_max"], det["y_max"]

            # Convert normalized to [x, y, w, h] pixel
            x1 = x1_pct * img_width
            y1 = y1_pct * img_height
            x2 = x2_pct * img_width
            y2 = y2_pct * img_height

            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            label = ""  # det.get('label', '')
            if det.get("overlapping_element_ids"):
                # Format list as string
                elem_ids = det["overlapping_element_ids"]
                if isinstance(elem_ids, list):
                    label += f" (elements: {elem_ids})"
                else:
                    label += f" (elements: {elem_ids})"

            boxes.append({"bbox": [x1, y1, w, h], "text": label, "number": i + 1, "color": colors[i % len(colors)]})

        # Draw numbered elements if provided
        if elements:
            for elem in elements:
                bbox = elem.get("bbox", [])
                if len(bbox) >= 4:
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

                    # Heuristic for normalized [x, y, w, h]
                    if w <= 1.0 and h <= 1.0 and x <= 1.0:
                        x *= img_width
                        y *= img_height
                        w *= img_width
                        h *= img_height

                    boxes.append(
                        {
                            "bbox": [x, y, w, h],
                            "number": elem.get("element_id"),
                            "color": "#FFFFFF",  # White dashed box for elements
                        }
                    )

        add_overlays_to_image(image_path, boxes, output_path=output_path)

        print(f"[AttentionExtractor] Saved visualization: {output_path}")
        return output_path

    def visualize_masks(
        self,
        image_path: str,
        masks: List[np.ndarray],
        output_path: str = "masks.png",
        alpha: float = 0.5,
        draw_ids: bool = False,
        colors: Optional[List[List[float]]] = None,
    ) -> str:
        """
        Visualize binary masks on top of the image.
        
        Args:
            image_path: Path to original image
            masks: List of binary masks (H, W) or (1, H, W)
            output_path: Where to save the result
            alpha: Transparency of masks (0.0 - 1.0)
            draw_ids: If True, draw the index number at the center of each mask
            colors: Optional list of RGB colors [0-255] or [0-1] for each mask.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as path_effects
        import random
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_array)
        
        def get_random_color():
            return [random.random(), random.random(), random.random()]
        
        indexed_masks = []
        for i, m in enumerate(masks):
            if len(m.shape) == 3: m = m[0]
            indexed_masks.append((i, m))
            
        sorted_masks = sorted(indexed_masks, key=lambda x: np.sum(x[1]), reverse=True)
        
        for idx, mask in sorted_masks:
            if len(mask.shape) != 2:
                continue

            if colors and idx < len(colors):
                c = np.array(colors[idx])
                if np.max(c) > 1.0:
                    c = c / 255.0
                color = np.concatenate([c, [alpha]])
            else:
                color = np.concatenate([get_random_color(), [alpha]])
            
            h, w = mask.shape
            mask_image = np.zeros((h, w, 4))
            mask_image[mask > 0] = color
            
            ax.imshow(mask_image)
            
            if draw_ids:
                import cv2
                h_m, w_m = mask.shape
                padded_mask = np.zeros((h_m + 2, w_m + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = mask.astype(np.uint8)
                
                dist = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)
                _, max_val, _, max_loc = cv2.minMaxLoc(dist)
                
                cx = max_loc[0] - 1
                cy = max_loc[1] - 1
                
                cx = max(0, min(w_m - 1, cx))
                cy = max(0, min(h_m - 1, cy))
                
                label_text = str(idx)
                font_size = 10
                padding = 1
                
                est_w = max(8, len(label_text) * 0.6 * font_size)
                est_h = font_size * 1.0
                
                box_w = est_w + (padding * 2)
                box_h = est_h + (padding * 2)
                
                x0 = cx - box_w/2
                y0 = cy - box_h/2
                
                rect = plt.Rectangle((x0, y0), box_w, box_h, 
                                   facecolor='black', 
                                   edgecolor='none', 
                                   linewidth=0,
                                   alpha=0.7,
                                   zorder=100)
                ax.add_patch(rect)
                
                txt = ax.text(cx, cy, label_text, 
                        color='white', 
                        fontsize=font_size, 
                        ha='center', 
                        va='center', 
                        fontweight='bold',
                        zorder=101)
            
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[AttentionExtractor] Saved mask visualization: {output_path}")
        return output_path

    def visualize_masks_simple(self, *args, **kwargs):
         return self.visualize_masks(*args, **kwargs)

    def import_patheffects_helper(self):
         import matplotlib.patheffects as path_effects
         return path_effects
