"""
AttentionExtractor - Stage 2 of the captcha solving pipeline.

Uses VLM pointing/detection capabilities to extract coordinates.
Supports multiple backends:
- moondream: Uses native point()/detect() methods
- qwen-vl: Uses Ollama with coordinate prompting

Enhanced with intelligent element detection and focus capabilities.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple, List, Literal, Dict, Any, TYPE_CHECKING
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch


class AttentionExtractor:
    """
    Extracts coordinates using VLM pointing/detection capabilities.
    
    Backends:
    - "moondream": Uses vikhyatk/moondream2 (HF Transformers) with native point()/detect() methods
    - "qwen-vl": Uses Qwen2-VL via Ollama with coordinate prompting
    
    Enhanced capabilities:
    - detect_objects(): Find all instances of a class (birds, cars, etc.)
    - detect_interactable_elements(): Automatically find clickable elements
    - focus(): High-precision pointing for specific elements
    - map_detections_to_elements(): Map detected objects to numbered elements
    """
    
    def __init__(
        self,
        model: str = "vikhyatk/moondream2",
        device: Optional[str] = None,
        backend: Literal["moondream", "qwen-vl"] = "moondream",
    ):
        self.model_id = model
        self.backend = backend
        
        # Decide device
        if device is None:
            import torch
            # Prefer CUDA/ROCm if available (torch reports ROCm as cuda)
            if torch.cuda.is_available():
                self.device = "cuda"
                if hasattr(torch.version, "hip") and torch.version.hip:
                    print(f"[AttentionExtractor] Detected ROCm (HIP {torch.version.hip}), using AMD GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._model = None
        
        # For visualization
        self._last_attention_grid = None
        self._last_grid_size = None
        self._last_points = None
        self._last_detections = None

    def _load_model(self):
        if self._model is not None:
            return
        
        if self.backend == "moondream":
            self._load_moondream()
        elif self.backend == "qwen-vl":
            # Qwen-VL uses Ollama, no local model to load
            pass
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _load_moondream(self):
        """Load moondream v2 via the Hugging Face implementation."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"[AttentionExtractor] Loading moondream: {self.model_id}")
        
        # Prefer FP16 on GPU; CPU stays FP32
        if self.device == "cuda":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self._model.eval()

        try:
            AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"[AttentionExtractor] AutoTokenizer unavailable: {e}")

        dtype_display = getattr(self._model, "dtype", dtype)
        print(f"[AttentionExtractor] Moondream loaded on {self.device} (dtype={dtype_display})")
    
    # =========================================================================
    # CORE EXTRACTION METHODS
    # =========================================================================
    
    def extract_coordinates(
        self,
        image_path: str,
        target_description: str,
    ) -> Tuple[float, float]:
        """
        Extract coordinates for a target element in the image.
        
        Args:
            image_path: Path to the image
            target_description: Description of what to find (e.g., "the checkbox")
        
        Returns:
            Tuple of (x, y) as percentages in [0.0, 1.0] range
        """
        self._load_model()
        
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size
        
        print(f"[AttentionExtractor] Image: {img_width}x{img_height}")
        print(f"[AttentionExtractor] Target: {target_description}")
        
        if self.backend == "moondream":
            x_pct, y_pct = self._extract_moondream(image, target_description)
        elif self.backend == "qwen-vl":
            x_pct, y_pct = self._extract_qwen_vl(image_path, target_description, img_width, img_height)
        else:
            x_pct, y_pct = 0.5, 0.5
        
        # Clamp to valid range
        x_pct = max(0.0, min(1.0, x_pct))
        y_pct = max(0.0, min(1.0, y_pct))
        
        # Store pixel coords for visualization
        self._last_points = [(int(x_pct * img_width), int(y_pct * img_height))]
        
        print(f"[AttentionExtractor] Result: ({x_pct:.4f}, {y_pct:.4f})")
        return (x_pct, y_pct)
    
    def focus(
        self,
        image_path: str,
        target_description: str,
    ) -> Tuple[float, float]:
        """
        High-precision pointing using moondream's point() method.
        
        This is an alias for extract_coordinates but emphasizes
        the use of point() for precision focusing on specific elements.
        
        Args:
            image_path: Path to the image
            target_description: Detailed description of what to focus on
        
        Returns:
            Tuple of (x, y) as percentages in [0.0, 1.0] range
        """
        return self.extract_coordinates(image_path, target_description)
    
    def detect_objects(
        self,
        image_path: str,
        object_class: str,
    ) -> List[Dict[str, Any]]:
        """
        Detect all instances of an object class in the image.
        
        Uses moondream's detect() method to find all matching objects.
        
        Args:
            image_path: Path to the image
            object_class: Class to detect (e.g., "bird", "car", "traffic light")
        
        Returns:
            List of detected objects with bounding boxes:
            [{"label": str, "bbox": [x_min, y_min, x_max, y_max], ...}, ...]
            Coordinates are percentages in [0.0, 1.0] range.
        """
        self._load_model()
        
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size
        
        print(f"[AttentionExtractor] Detecting: {object_class}")
        
        detections = []
        
        if self.backend == "moondream" and hasattr(self._model, 'detect'):
            import torch
            with torch.no_grad():
                try:
                    # Try preferred signature from Moondream3 preview: detect(image=..., query=..., settings)
                    result = None
                    try:
                        result = self._model.detect(
                            image=image,
                            query=object_class,
                            settings={"max_objects": 12},
                        )
                    except TypeError:
                        # Fallback to positional signature detect(image, "car")
                        result = self._model.detect(image, object_class)
                    
                    # detect() returns {'objects': [{'x_min', 'y_min', 'x_max', 'y_max'}, ...]}
                    if result and 'objects' in result:
                        for obj in result['objects']:
                            detections.append({
                                "label": object_class,
                                "bbox": [obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max']],
                            })
                        print(f"[AttentionExtractor] detect() found {len(detections)} {object_class}(s)")
                except Exception as e:
                    print(f"[AttentionExtractor] detect() failed: {e}")
        
        self._last_detections = detections
        return detections
    
    def detect_interactable_elements(
        self,
        image_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Automatically detect interactable elements like buttons, checkboxes, images.
        
        Uses moondream to identify clickable/draggable areas.
        
        Args:
            image_path: Path to the image
        
        Returns:
            List of detected elements with bounding boxes
        """
        self._load_model()
        
        image = Image.open(image_path).convert("RGB")
        
        all_elements = []
        
        # Try to detect common interactable element types
        element_types = ["button", "checkbox", "image", "icon", "tile"]
        
        if self.backend == "moondream" and hasattr(self._model, 'detect'):
            import torch
            with torch.no_grad():
                for elem_type in element_types:
                    try:
                        result = self._model.detect(image, elem_type)
                        if result and 'objects' in result:
                            for obj in result['objects']:
                                all_elements.append({
                                    "label": elem_type,
                                    "bbox": [obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max']],
                                    "element_type": elem_type,
                                })
                    except Exception:
                        continue
        
        print(f"[AttentionExtractor] Found {len(all_elements)} interactable elements")
        return all_elements
    
    def map_detections_to_elements(
        self,
        detections: List[Dict[str, Any]],
        elements: List[Dict[str, Any]],
        iou_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Map detected objects to numbered elements based on bounding box overlap.
        
        Args:
            detections: List of detections from detect_objects()
            elements: List of numbered elements (with 'element_id' and 'bbox')
            iou_threshold: Minimum IoU to consider a match
        
        Returns:
            Detections with 'overlapping_element_ids' field added
        """
        def calc_iou(box1, box2):
            """Calculate IoU between two boxes [x_min, y_min, x_max, y_max]."""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        def bbox_to_corners(bbox):
            """Convert [x, y, w, h] to [x_min, y_min, x_max, y_max]."""
            if len(bbox) == 4:
                # Check if already corners (values normalized 0-1)
                if bbox[2] <= 1.0 and bbox[3] <= 1.0:
                    return bbox  # Already in corner format
                # Otherwise it's [x, y, w, h] format
                return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            return bbox
        
        for det in detections:
            det_bbox = det['bbox']
            overlapping_ids = []
            
            for elem in elements:
                elem_bbox = bbox_to_corners(elem.get('bbox', []))
                if elem_bbox:
                    iou = calc_iou(det_bbox, elem_bbox)
                    if iou >= iou_threshold:
                        overlapping_ids.append(elem.get('element_id'))
            
            det['overlapping_element_ids'] = overlapping_ids
        
        return detections
    
    # =========================================================================
    # INTELLIGENT QUERIES
    # =========================================================================
    
    def describe_element(
        self,
        image_path: str,
        element_description: str,
    ) -> str:
        """
        Get a detailed description of an element in the image.
        
        Useful for understanding what an element is before deciding to interact.
        
        Args:
            image_path: Path to the image
            element_description: What to describe (e.g., "element 3", "the draggable piece")
        
        Returns:
            Text description of the element
        """
        self._load_model()
        
        image = Image.open(image_path).convert("RGB")
        
        if self.backend == "moondream":
            import torch
            with torch.no_grad():
                if hasattr(self._model, 'query'):
                    # Use moondream's query method if available
                    prompt = f"Describe {element_description} in detail. What does it look like?"
                    try:
                        result = self._model.query(image, prompt)
                        if isinstance(result, dict):
                            return result.get('answer', str(result))
                        return str(result)
                    except Exception as e:
                        print(f"[AttentionExtractor] query() failed: {e}")
                
                # Fallback: use caption with encoding
                if hasattr(self._model, 'caption'):
                    try:
                        result = self._model.caption(image)
                        return result.get('caption', '') if isinstance(result, dict) else str(result)
                    except Exception:
                        pass
        
        return f"Unable to describe {element_description}"
    
    def ask_about_image(
        self,
        image_path: str,
        question: str,
    ) -> str:
        """
        Ask a question about the image.
        
        Args:
            image_path: Path to the image
            question: Question to ask (e.g., "What should be dragged?", "Which elements contain birds?")
        
        Returns:
            Answer to the question
        """
        self._load_model()
        
        image = Image.open(image_path).convert("RGB")
        
        if self.backend == "moondream":
            import torch
            with torch.no_grad():
                if hasattr(self._model, 'query'):
                    try:
                        result = self._model.query(image, question)
                        if isinstance(result, dict):
                            return result.get('answer', str(result))
                        return str(result)
                    except Exception as e:
                        print(f"[AttentionExtractor] query() failed: {e}")
        
        elif self.backend == "qwen-vl":
            return self._ask_qwen_vl(image_path, question)
        
        return "Unable to answer the question"
    
    def _ask_qwen_vl(self, image_path: str, question: str) -> str:
        """Ask a question using Ollama."""
        import ollama
        
        ollama_model = self.model_id if self.model_id else "hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M"
        
        try:
            response = ollama.chat(
                model=ollama_model,
                messages=[
                    {
                        "role": "user",
                        "content": question,
                        "images": [image_path]
                    }
                ],
                options={"temperature": 0.1}
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"[AttentionExtractor] Ollama failed: {e}")
            return "Unable to answer"

    # =========================================================================
    # BACKEND-SPECIFIC EXTRACTION METHODS
    # =========================================================================
    
    def _extract_moondream(self, image: Image.Image, target_description: str) -> Tuple[float, float]:
        """Extract coordinates using moondream's native point() method or manual generation."""
        import torch
        
        with torch.no_grad():
            # Use moondream's native point() method if available (original Moondream models)
            if hasattr(self._model, 'point'):
                print("[AttentionExtractor] Using moondream point() method")
                try:
                    result = self._model.point(image, target_description)
                    if result and 'points' in result and len(result['points']) > 0:
                        point = result['points'][0]
                        x_pct, y_pct = point['x'], point['y']
                        print(f"[AttentionExtractor] point() returned: ({x_pct:.4f}, {y_pct:.4f})")
                        return (x_pct, y_pct)
                    else:
                        print(f"[AttentionExtractor] point() returned no points: {result}")
                except Exception as e:
                    print(f"[AttentionExtractor] point() failed: {e}")
            
            # Try detect() method as fallback (original Moondream models)
            elif hasattr(self._model, 'detect'):
                print("[AttentionExtractor] Using moondream detect() method")
                try:
                    result = self._model.detect(image, target_description)
                    if result and 'objects' in result and len(result['objects']) > 0:
                        obj = result['objects'][0]
                        x_pct = (obj['x_min'] + obj['x_max']) / 2
                        y_pct = (obj['y_min'] + obj['y_max']) / 2
                        print(f"[AttentionExtractor] detect() found box, center: ({x_pct:.4f}, {y_pct:.4f})")
                        return (x_pct, y_pct)
                    else:
                        print(f"[AttentionExtractor] detect() returned no objects: {result}")
                except Exception as e:
                    print(f"[AttentionExtractor] detect() failed: {e}")

            # Fallback to center if nothing works
            print("[AttentionExtractor] No detection methods worked, using center")
            return (0.5, 0.5)
    
    def _extract_qwen_vl(
        self,
        image_path: str,
        target_description: str,
        img_width: int,
        img_height: int
    ) -> Tuple[float, float]:
        """Extract coordinates using a multimodal model via Ollama with coordinate prompting."""
        import ollama
        
        # Use model_id for Ollama model name
        ollama_model = self.model_id if self.model_id else "hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M"
        
        # Prompt the model to output percentage coordinates
        prompt = f"""Look at this image carefully. I need to click on {target_description}.

Tell me the position as a percentage of image width and height.
x=0% is the left edge, x=100% is the right edge.
y=0% is the top edge, y=100% is the bottom edge.

Respond with ONLY the coordinates in this exact format: x=NUMBER%, y=NUMBER%

For example: x=25%, y=50%"""

        try:
            print(f"[AttentionExtractor] Using Ollama model: {ollama_model}")
            response = ollama.chat(
                model=ollama_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_path]
                    }
                ],
                options={"temperature": 0.1}
            )
            
            text = response["message"]["content"]
            print(f"[AttentionExtractor] Ollama response: {text}")
            
            return self._parse_percentage_from_text(text)
            
        except Exception as e:
            print(f"[AttentionExtractor] Ollama failed: {e}")
            return (0.5, 0.5)
    
    def _parse_percentage_from_text(self, response: str) -> Tuple[float, float]:
        """Parse x, y percentages from text response."""
        
        # Try various patterns for percentage format
        patterns = [
            r'x\s*=\s*(\d+(?:\.\d+)?)\s*%\s*,?\s*y\s*=\s*(\d+(?:\.\d+)?)\s*%',  # x=25%, y=50%
            r'x\s*:\s*(\d+(?:\.\d+)?)\s*%\s*,?\s*y\s*:\s*(\d+(?:\.\d+)?)\s*%',  # x: 25%, y: 50%
            r'\((\d+(?:\.\d+)?)\s*%\s*,\s*(\d+(?:\.\d+)?)\s*%\)',                # (25%, 50%)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                x_pct = float(match.group(1)) / 100.0
                y_pct = float(match.group(2)) / 100.0
                if 0 <= x_pct <= 1 and 0 <= y_pct <= 1:
                    print(f"[AttentionExtractor] Parsed percentages: ({x_pct:.4f}, {y_pct:.4f})")
                    return (x_pct, y_pct)
        
        # Fallback to center
        print(f"[AttentionExtractor] Could not parse percentages, using center")
        return (0.5, 0.5)
    
    # =========================================================================
    # DRAG OPERATIONS
    # =========================================================================
    
    def extract_drag_coordinates(
        self,
        image_path: str,
        source_description: str,
        target_description: str,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Extract coordinates for drag operation (source and target) as percentages."""
        source = self.extract_coordinates(image_path, source_description)
        target = self.extract_coordinates(image_path, target_description)
        return (source, target)
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def visualize_attention(
        self,
        image_path: str,
        target_description: str,
        output_path: str = "attention_heatmap.png",
        show_peak: bool = True
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
            ax.axhline(y=py, color='lime', linewidth=1, alpha=0.5)
            ax.axvline(x=px, color='lime', linewidth=1, alpha=0.5)
            # Draw marker
            ax.scatter([px], [py], c='lime', s=200, marker='x', linewidths=3, zorder=10)
            ax.add_patch(plt.Circle((px, py), 20, fill=False, color='lime', linewidth=2, zorder=10))
        
        ax.set_title(f'Target: "{target_description}"\nPosition: ({x_pct:.2%}, {y_pct:.2%}) | Pixel: ({px}, {py})')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)
        img_height, img_width = img_array.shape[:2]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_array)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            color = colors[i % len(colors)]
            
            # Convert from percentage to pixel coordinates
            x1 = bbox[0] * img_width
            y1 = bbox[1] * img_height
            x2 = bbox[2] * img_width
            y2 = bbox[3] * img_height
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            label = det.get('label', '')
            if det.get('overlapping_element_ids'):
                label += f" (elements: {det['overlapping_element_ids']})"
            
            ax.text(x1, y1 - 5, label, color=color, fontsize=10, fontweight='bold')
        
        # Draw numbered elements if provided
        if elements:
            for elem in elements:
                bbox = elem.get('bbox', [])
                if len(bbox) >= 4:
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    # Determine if bbox is percentage or pixel based
                    if w <= 1 and h <= 1:  # Percentage
                        x, y = x * img_width, y * img_height
                        w, h = w * img_width, h * img_height
                    
                    rect = patches.Rectangle(
                        (x, y), w, h,
                        linewidth=1, edgecolor='white', facecolor='none',
                        linestyle='--'
                    )
                    ax.add_patch(rect)
                    ax.text(x + 2, y + 12, str(elem.get('element_id', '')), 
                           color='white', fontsize=8, fontweight='bold')
        
        ax.set_title(f'Detected {len(detections)} objects')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[AttentionExtractor] Saved visualization: {output_path}")
        return output_path
    
    def get_attention_heatmap(self, image_path: str, target_description: str) -> Optional[np.ndarray]:
        """Get attention heatmap if available (for visualization)."""
        self.extract_coordinates(image_path, target_description)
        return self._last_attention_grid
