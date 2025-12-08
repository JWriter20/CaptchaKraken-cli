"""
AttentionExtractor - Minimal wrapper around moondream's point() and detect() methods.

Exposes:
- point(): Find coordinates of a target element
- detect(): Find all instances of an object class
- clusterDetections(): Merge overlapping/close detection boxes
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
    """Minimal wrapper around moondream for pointing and detection."""

    def __init__(
        self,
        model: str = "vikhyatk/moondream2",
        device: Optional[str] = None,
    ):
        self.model_id = model

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

        self._model = None

    def _load_model(self):
        """Lazy load moondream model."""
        if self._model is not None:
            return

        t0 = time.time()
        import torch
        from transformers import AutoModelForCausalLM

        print(f"[AttentionExtractor] Loading {self.model_id}...", file=sys.stderr)

        if self.device in ("cuda", "mps"):
            dtype = torch.float16
        else:
            dtype = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self._model.eval()

        t1 = time.time()
        print(f"[AttentionExtractor] Loaded on {self.device} in {t1 - t0:.2f}s", file=sys.stderr)

    def point(
        self,
        image_path: str,
        target: str,
    ) -> Dict[str, Any]:
        """
        Find coordinates of target element using moondream's point().

        Args:
            image_path: Path to image
            target: What to find (e.g., "the checkbox")

        Returns:
            {'points': [{'x': float, 'y': float}, ...]}
            Coordinates are percentages in [0.0, 1.0]
        """
        self._load_model()

        t0 = time.time()
        image = Image.open(image_path).convert("RGB")

        import torch

        with torch.no_grad():
            if hasattr(self._model, "point"):
                result = self._model.point(image, target)
                t1 = time.time()
                print(f"[AttentionExtractor] point('{target}') took {t1 - t0:.2f}s", file=sys.stderr)
                return result if result else {"points": []}
            else:
                print("[AttentionExtractor] Model doesn't have point() method", file=sys.stderr)
                return {"points": []}

    def detect(
        self,
        image_path: str,
        object_class: str,
        max_objects: int = 24,
    ) -> Dict[str, Any]:
        """
        Detect all instances of object class using moondream's detect().

        Args:
            image_path: Path to image
            object_class: What to detect (e.g., "bird", "car")
            max_objects: Maximum objects to detect

        Returns:
            {'objects': [{'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float}, ...]}
            Coordinates are percentages in [0.0, 1.0]
        """
        self._load_model()

        t0 = time.time()
        image = Image.open(image_path).convert("RGB")

        import torch

        with torch.no_grad():
            if hasattr(self._model, "detect"):
                try:
                    # Try keyword args first
                    result = self._model.detect(
                        image=image,
                        query=object_class,
                        settings={"max_objects": max_objects},
                    )
                except TypeError:
                    # Fallback to positional args
                    result = self._model.detect(image, object_class)

                t1 = time.time()
                print(f"[AttentionExtractor] detect('{object_class}') took {t1 - t0:.2f}s", file=sys.stderr)
                return result if result else {"objects": []}
            else:
                print("[AttentionExtractor] Model doesn't have detect() method", file=sys.stderr)
                return {"objects": []}
                
    def ask(
        self,
        image_path: str,
        question: str,
    ) -> str:
        """
        Ask a question about the image using moondream.

        Args:
            image_path: Path to image
            question: Question to ask

        Returns:
            Answer string
        """
        self._load_model()

        t0 = time.time()
        image = Image.open(image_path).convert("RGB")

        import torch

        with torch.no_grad():
            if hasattr(self._model, "query"):
                ans = self._model.query(image, question)["answer"]
                t1 = time.time()
                print(f"[AttentionExtractor] ask('{question[:30]}...') took {t1 - t0:.2f}s", file=sys.stderr)
                return ans
            elif hasattr(self._model, "answer_question"):
                # Some versions require encoding first
                enc_image = self._model.encode_image(image)
                ans = self._model.answer_question(enc_image, question)
                t1 = time.time()
                print(f"[AttentionExtractor] ask('{question[:30]}...') took {t1 - t0:.2f}s", file=sys.stderr)
                return ans
            else:
                print("[AttentionExtractor] Model doesn't have query() or answer_question() method", file=sys.stderr)
                return ""

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
