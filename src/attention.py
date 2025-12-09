"""
AttentionExtractor - Wrapper around GroundingDINO and SAM 2.

Exposes:
- point(): Find coordinates of a target element (center of detected box)
- detect(): Find all instances of an object class (with sensitivity adjustment)
- clusterDetections(): Merge overlapping/close detection boxes
- detectInteractable(): Detect interactable objects using SAM 2 segmentation and heuristic filtering
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    pass


def clusterDetections(
    detections: List[Dict[str, float]],
    iou_threshold: float = 0.5,
    distance_threshold: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    Cluster and merge overlapping or very close detection bounding boxes.

    Can use either IoU-based clustering (for overlapping boxes) or
    distance-based clustering (for nearby but non-overlapping boxes).

    Args:
        detections: List of bounding boxes with keys: x_min, y_min, x_max, y_max
        iou_threshold: IoU threshold for merging (default: 0.5)
        distance_threshold: If set, use center distance instead of IoU.
                          Boxes with centers closer than this are merged.
                          (Recommended: 0.1-0.3 for normalized coordinates)

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

    def average_boxes(boxes: List[Dict[str, float]]) -> Dict[str, float]:
        """Average coordinates of multiple boxes."""
        n = len(boxes)
        return {
            "x_min": sum(b["x_min"] for b in boxes) / n,
            "y_min": sum(b["y_min"] for b in boxes) / n,
            "x_max": sum(b["x_max"] for b in boxes) / n,
            "y_max": sum(b["y_max"] for b in boxes) / n,
        }

    # Greedy clustering: merge boxes based on IoU or distance
    clusters = []
    used = [False] * len(detections)

    # Choose similarity function based on parameters
    if distance_threshold is not None:
        # Use center distance
        def should_merge(box1: Dict[str, float], box2: Dict[str, float]) -> bool:
            return calculate_center_distance(box1, box2) < distance_threshold
    else:
        # Use IoU
        def should_merge(box1: Dict[str, float], box2: Dict[str, float]) -> bool:
            return calculate_iou(box1, box2) > iou_threshold

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

    # Average each cluster into a single box
    merged = [average_boxes(cluster) for cluster in clusters]

    return merged


class AttentionExtractor:
    """Wrapper around GroundingDINO and SAM 2 for pointing and detection."""

    def __init__(
        self,
        device: Optional[str] = None,
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

        self._gd_model = None
        self._gd_processor = None
        self._sam2_model = None
        self._sam2_processor = None

    def unload_models(self, models: Optional[List[str]] = None):
        """
        Unload specific models or all models to free memory.
        
        Args:
            models: List of model names to unload ("sam2", "gd").
                   If None, unloads all.
        """
        import gc
        import torch

        all_models = {
            "sam2": ["_sam2_model", "_sam2_processor"],
            "gd": ["_gd_model", "_gd_processor"],
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

    def detectInteractable(
        self,
        image_path: str,
        points_per_side: int = 8,
        score_threshold: float = 0.3,
        max_area_ratio: float = 0.4,
        aspect_ratio_threshold: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Detect interactable objects using SAM 2 segmentation and heuristic filtering.
        
        Args:
            image_path: Path to image
            points_per_side: Density of point grid for SAM 2 (default: 8, high sensitivity)
            score_threshold: Minimum IoU score for SAM 2 masks (default: 0.3, high sensitivity)
            max_area_ratio: Maximum fraction of image area a mask can cover
            aspect_ratio_threshold: Maximum allowed aspect ratio (longer side / shorter side)
            
        Returns:
            {'objects': [{'bbox': [x, y, w, h], 'area': float, 'solidity': float, ...}]}
            Bounding boxes are in pixels.
        """
        # Run SAM 2 with automatic grid prompts
        sam_result = self.segment_with_sam2(
            image_path,
            points_per_side=points_per_side,
            score_threshold=score_threshold
        )
        
        masks = sam_result.get("masks", [])
        if not masks:
            return {"objects": []}
            
        with Image.open(image_path) as img:
            img_w, img_h = img.size
            
        total_area = img_w * img_h
        valid_objects = []
        
        for i, mask in enumerate(masks):
            if not isinstance(mask, np.ndarray):
                continue
                
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            x1, x2 = np.min(x_indices), np.max(x_indices)
            y1, y2 = np.min(y_indices), np.max(y_indices)
            w = x2 - x1
            h = y2 - y1
            
            # Filter tiny things
            if w < 10 or h < 10:
                continue
                
            area = w * h
            
            # Filter massive things (background)
            if area > (total_area * max_area_ratio):
                # print(f"  - Dropping mask {i} (too large)")
                continue
            
            # Filter long skinny things (extreme aspect ratio)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999
            if aspect_ratio > aspect_ratio_threshold:
                # print(f"  - Dropping mask {i} (extreme aspect ratio: {w}x{h}, ratio={aspect_ratio:.2f})")
                continue
                
            valid_objects.append({
                "bbox": [int(x1), int(y1), int(w), int(h)],
                "area": int(area),
                "aspect_ratio": float(aspect_ratio),
                "mask_index": i
            })
            
        # Sort by area descending
        valid_objects.sort(key=lambda x: x["area"], reverse=True)
        
        # Filter duplicates/overlaps
        final_objects = []
        
        def compute_iou_bbox(b1, b2):
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[0]+b1[2], b2[0]+b2[2])
            y2 = min(b1[1]+b1[3], b2[1]+b2[3])
            
            w_inter = max(0, x2 - x1)
            h_inter = max(0, y2 - y1)
            inter = w_inter * h_inter
            
            area1 = b1[2] * b1[3]
            area2 = b2[2] * b2[3]
            union = area1 + area2 - inter
            
            return inter / union if union > 0 else 0
            
        for obj in valid_objects:
            is_overlap = False
            for k in final_objects:
                if compute_iou_bbox(obj["bbox"], k["bbox"]) > 0.6: 
                    is_overlap = True
                    break
            if not is_overlap:
                final_objects.append(obj)
                
        return {"objects": final_objects}

    def point(
        self,
        image_path: str,
        target: str,
    ) -> Dict[str, Any]:
        """
        Find coordinates of target element using GroundingDINO.
        Returns the center of the detected bounding box.

        Args:
            image_path: Path to image
            target: What to find (e.g., "the checkbox")

        Returns:
            {'points': [{'x': float, 'y': float}, ...]}
            Coordinates are percentages in [0.0, 1.0]
        """
        # Use detect (which has sensitivity loop)
        result = self.detect(image_path, target, max_objects=1)
        
        if result.get("objects"):
            obj = result["objects"][0]
            # Calculate center
            center_x = (obj["x_min"] + obj["x_max"]) / 2
            center_y = (obj["y_min"] + obj["y_max"]) / 2
            return {"points": [{"x": center_x, "y": center_y}]}
        
        return {"points": []}

    def detect(
        self,
        image_path: str,
        object_class: str,
        max_objects: int = 24,
    ) -> Dict[str, Any]:
        """
        Detect all instances of object class using GroundingDINO.
        
        Automatically adjusts sensitivity (thresholds) if no objects are found.

        Args:
            image_path: Path to image
            object_class: What to detect (e.g., "bird", "car")
            max_objects: Maximum objects to detect

        Returns:
            {'objects': [{'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float}, ...]}
            Coordinates are percentages in [0.0, 1.0]
        """
        # Sensitivity loop: try progressively lower thresholds
        thresholds = [
            (0.35, 0.25),  # Default
            (0.30, 0.25),
            (0.25, 0.20),
            (0.20, 0.15),
            (0.15, 0.10),
            (0.10, 0.05),  # Very sensitive
        ]

        for i, (box_th, text_th) in enumerate(thresholds):
            result = self.detect_with_grounding_dino(
                image_path, 
                object_class, 
                box_threshold=box_th, 
                text_threshold=text_th
            )
            
            if result.get("objects"):
                if i > 0:
                    print(f"[AttentionExtractor] Detected '{object_class}' at sensitivity level {i+1} (box_th={box_th}, text_th={text_th})", file=sys.stderr)
                
                # Sort by score descending
                objects = sorted(result["objects"], key=lambda x: x.get("score", 0), reverse=True)
                return {"objects": objects[:max_objects]}
        
        print(f"[AttentionExtractor] No objects found for '{object_class}' even at highest sensitivity.", file=sys.stderr)
        return {"objects": []}

    def _load_grounding_dino(self):
        """Lazy load GroundingDINO model."""
        if self._gd_model is not None:
            return

        t0 = time.time()
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        model_id = "IDEA-Research/grounding-dino-tiny"
        print(f"[AttentionExtractor] Loading {model_id}...", file=sys.stderr)
        
        self._gd_processor = AutoProcessor.from_pretrained(model_id)
        self._gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self._gd_model.eval()
        
        t1 = time.time()
        print(f"[AttentionExtractor] Loaded GroundingDINO in {t1 - t0:.2f}s", file=sys.stderr)

    def _load_sam2(self):
        """Lazy load SAM 2 model."""
        if self._sam2_model is not None:
            return

        t0 = time.time()
        import torch
        from transformers import Sam2Model, Sam2Processor

        model_id = "facebook/sam2-hiera-large"
        print(f"[AttentionExtractor] Loading {model_id}...", file=sys.stderr)

        self._sam2_processor = Sam2Processor.from_pretrained(model_id)
        self._sam2_model = Sam2Model.from_pretrained(model_id).to(self.device)
        self._sam2_model.eval()

        t1 = time.time()
        print(f"[AttentionExtractor] Loaded SAM 2 in {t1 - t0:.2f}s", file=sys.stderr)

    def detect_with_grounding_dino(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Detect objects using GroundingDINO.
        
        Args:
            image_path: Path to image
            text_prompt: Text description of objects (e.g. "a cat. a dog.")
            box_threshold: Confidence threshold for boxes
            text_threshold: Confidence threshold for text
            
        Returns:
            {'objects': [{'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float, 'label': str, 'score': float}, ...]}
        """
        self._load_grounding_dino()
        import torch

        t0 = time.time()
        image = Image.open(image_path).convert("RGB")
        
        inputs = self._gd_processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._gd_model(**inputs)

        results = self._gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )[0]
        
        objects = []
        width, height = image.size
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # box is [x_min, y_min, x_max, y_max] in pixels
            x_min, y_min, x_max, y_max = box.tolist()
            objects.append({
                "x_min": x_min / width,
                "y_min": y_min / height,
                "x_max": x_max / width,
                "y_max": y_max / height,
                "score": score.item(),
                "label": label
            })
            
        t1 = time.time()
        # print(f"[AttentionExtractor] GroundingDINO detect('{text_prompt}') took {t1 - t0:.2f}s", file=sys.stderr)
        return {"objects": objects}

    def segment_with_sam2(
        self,
        image_path: str,
        input_boxes: Optional[List[List[float]]] = None,
        input_points: Optional[List[List[float]]] = None,
        points_per_side: Optional[int] = None,
        score_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Segment objects using SAM 2.
        
        Args:
            image_path: Path to image
            input_boxes: List of [x_min, y_min, x_max, y_max] (normalized 0-1)
            points_per_side: If no prompts provided, use this many points per side for grid (default: 6)
            score_threshold: Minimum IoU score to keep a mask (default: 0.0)
            
        Returns:
            {'masks': List[numpy.ndarray]} - Binary masks (H, W)
        """
        self._load_sam2()
        import torch

        t0 = time.time()
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        inputs_kwargs = {}
        
        if input_boxes:
            # Scale boxes to pixels
            # SAM 2 processor expects input_boxes shape: (batch_size, num_boxes, 4)
            # For single image: input_boxes=[[[x1, y1, x2, y2], [x3, y3, x4, y4], ...]]
            pixel_boxes = [[
                b[0] * width, b[1] * height, b[2] * width, b[3] * height
            ] for b in input_boxes]
            
            inputs_kwargs["input_boxes"] = [pixel_boxes]

        if not inputs_kwargs:
             # If no prompts provided, generate grid of points for automatic segmentation
             print("[AttentionExtractor] No prompts provided, generating grid of points...", file=sys.stderr)
             
             # Generate grid of points
             n_points = points_per_side if points_per_side else 6
             x = np.linspace(0, width, n_points)
             y = np.linspace(0, height, n_points)
             xv, yv = np.meshgrid(x, y)
             points = np.stack([xv.flatten(), yv.flatten()], axis=1)
             
             # We want each point to be a SEPARATE prompt.
             # Shape: (batch_size=1, num_prompts=N, num_points_per_prompt=1, 2)
             
             reshaped_points = points.reshape(len(points), 1, 2)
             inputs_kwargs["input_points"] = [reshaped_points.tolist()]
             
             inputs_kwargs["input_labels"] = [[[1]] * len(points)]
             
             print(f"[AttentionExtractor] Generated {len(points)} point prompts ({n_points}x{n_points}).", file=sys.stderr)

        try:
            # Clear cache before heavy operation
            if self.device == "mps":
                import torch
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

            inputs = self._sam2_processor(images=image, return_tensors="pt", **inputs_kwargs)
            
            # Move to device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if self.device == "mps" and v.dtype == torch.float64:
                        v = v.to(torch.float32)
                    inputs[k] = v.to(self.device)

            with torch.no_grad():
                outputs = self._sam2_model(**inputs)
            
            reshaped_input_sizes = inputs.get("reshaped_input_sizes")
            if reshaped_input_sizes is None:
                reshaped_input_sizes = inputs["original_sizes"]

            try:
                masks = self._sam2_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    reshaped_input_sizes.cpu()
                )[0]
            except RuntimeError as re:
                # If dimension mismatch, try manual resize
                pred_masks = outputs.pred_masks[0] # (num_boxes, num_masks, H, W)
                orig_h, orig_w = inputs["original_sizes"][0].tolist()
                
                import torch.nn.functional as F
                num_boxes, num_masks, h, w = pred_masks.shape
                
                flat_masks = pred_masks.reshape(1, num_boxes * num_masks, h, w)
                
                resized_masks = F.interpolate(
                    flat_masks,
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False
                )
                
                masks = resized_masks.reshape(num_boxes, num_masks, orig_h, orig_w)
                masks = masks > 0.0

            scores = outputs.iou_scores.cpu() # (batch, num_boxes, num_masks)
            best_mask_indices = torch.argmax(scores, dim=2) # (batch, num_boxes)
            
            final_masks = []
            batch_idx = 0 
            
            num_queries = masks.shape[0]
            
            for i in range(num_queries):
                if i < best_mask_indices.shape[1]:
                    best_idx = best_mask_indices[batch_idx, i].item()
                    
                    if score_threshold > 0:
                        score = scores[batch_idx, i, best_idx].item()
                        if score < score_threshold:
                            continue
                            
                    mask = masks[i][best_idx].cpu().numpy() # (H, W) boolean
                    final_masks.append(mask)

            t1 = time.time()
            print(f"[AttentionExtractor] SAM 2 segmentation took {t1 - t0:.2f}s", file=sys.stderr)
            
            return {"masks": final_masks}

        except Exception as e:
            print(f"[AttentionExtractor] SAM 2 error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return {"masks": []}

    # =========================================================================
    # COMPATIBILITY METHODS
    # =========================================================================

    def detect_objects(
        self,
        image_path: str,
        object_class: str,
        max_objects: int = 24,
    ) -> List[Dict[str, Any]]:
        """Wrapper for detect() that returns the list of detections directly."""
        result = self.detect(image_path, object_class, max_objects=max_objects)
        detections = []
        if result and "objects" in result:
            for obj in result["objects"]:
                detections.append(
                    {
                        "label": object_class,
                        "bbox": [obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]],
                    }
                )
        return detections

    def extract_coordinates(
        self,
        image_path: str,
        target_description: str,
    ) -> Tuple[float, float]:
        """Wrapper for point() that returns a single (x, y) tuple."""
        result = self.point(image_path, target_description)
        if result and "points" in result and len(result["points"]) > 0:
            p = result["points"][0]
            return (p["x"], p["y"])
        return (0.5, 0.5)

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
        """Generate visualization showing detected point."""
        import matplotlib.pyplot as plt

        x_pct, y_pct = self.extract_coordinates(image_path, target_description)

        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)
        img_height, img_width = img_array.shape[:2]

        # Convert percentage to pixels for visualization
        px = int(x_pct * img_width)
        py = int(y_pct * img_height)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.imshow(img_array)

        if show_peak:
            # Draw crosshair
            ax.axhline(y=py, color="lime", linewidth=1, alpha=0.5)
            ax.axvline(x=px, color="lime", linewidth=1, alpha=0.5)
            # Draw marker
            ax.scatter([px], [py], c="lime", s=200, marker="x", linewidths=3, zorder=10)
            ax.add_patch(plt.Circle((px, py), 20, fill=False, color="lime", linewidth=2, zorder=10))

        ax.set_title(f'Target: "{target_description}"\nPosition: ({x_pct:.2%}, {y_pct:.2%}) | Pixel: ({px}, {py})')
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
        from .overlay import add_overlays_to_image

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
            bbox = det["bbox"]
            # Convert [x_min, y_min, x_max, y_max] normalized to [x, y, w, h] pixel
            x1 = bbox[0] * img_width
            y1 = bbox[1] * img_height
            x2 = bbox[2] * img_width
            y2 = bbox[3] * img_height

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
    ) -> str:
        """
        Visualize binary masks on top of the image.
        
        Args:
            image_path: Path to original image
            masks: List of binary masks (H, W) or (1, H, W)
            output_path: Where to save the result
            alpha: Transparency of masks (0.0 - 1.0)
        """
        import matplotlib.pyplot as plt
        import random
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_array)
        
        def get_random_color():
            return [random.random(), random.random(), random.random()]
        
        # Sort masks by area (largest first) so small ones aren't hidden
        sorted_masks = sorted(masks, key=lambda m: np.sum(m), reverse=True)
        
        for mask in sorted_masks:
            # Debug shape
            if hasattr(mask, "shape"):
                 pass # print(f"Processing mask with shape: {mask.shape}")
            
            # Ensure mask is 2D
            if len(mask.shape) == 3:
                mask = mask[0]
            elif len(mask.shape) != 2:
                print(f"[AttentionExtractor] Warning: Skipping mask with invalid shape {mask.shape}", file=sys.stderr)
                continue

            color = np.concatenate([get_random_color(), [alpha]])
            
            # Create colored mask
            h, w = mask.shape
            mask_image = np.zeros((h, w, 4))
            mask_image[mask > 0] = color
            
            ax.imshow(mask_image)
            
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[AttentionExtractor] Saved mask visualization: {output_path}")
        return output_path
