"""
AttentionExtractor - Stage 2 of the captcha solving pipeline.

Uses VLM pointing/detection capabilities to extract coordinates.
Supports multiple backends:
- moondream: Uses native point()/detect() methods
- qwen-vl: Uses Ollama with coordinate prompting
- florence: Uses Florence-2 grounding (requires GPU)
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple, List, Literal, TYPE_CHECKING
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch


class AttentionExtractor:
    """
    Extracts coordinates using VLM pointing/detection capabilities.
    
    Backends:
    - "moondream": Uses vikhyatk/moondream2 with native point()/detect() methods
    - "qwen-vl": Uses Qwen2-VL via Ollama with coordinate prompting
    - "florence": Uses Microsoft Florence-2 grounding
    """
    
    def __init__(
        self,
        model: str = "vikhyatk/moondream2",
        device: Optional[str] = None,
        backend: Literal["moondream", "qwen-vl", "florence"] = "moondream",
    ):
        self.model_id = model
        self.backend = backend
        
        self._model = None
        self._tokenizer = None
        self._processor = None
        
        # For visualization
        self._last_attention_grid = None
        self._last_grid_size = None
        self._last_points = None
        
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
    
    def _load_model(self):
        if self._model is not None:
            return
        
        if self.backend == "moondream":
            self._load_moondream()
        elif self.backend == "qwen-vl":
            # Qwen-VL uses Ollama, no local model to load
            pass
        elif self.backend == "florence":
            self._load_florence()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _load_moondream(self):
        """Load moondream2 model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"[AttentionExtractor] Loading moondream: {self.model_id}")
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self._model.eval()
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        
        print(f"[AttentionExtractor] Moondream loaded on {self.device}")
    
    def _load_florence(self):
        """Load Florence-2 model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        model_id = "microsoft/Florence-2-base"
        print(f"[AttentionExtractor] Loading Florence-2: {model_id}")
        
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self._model.eval()
        
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        print(f"[AttentionExtractor] Florence-2 loaded on {self.device}")
    
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
        elif self.backend == "florence":
            x_pct, y_pct = self._extract_florence(image, target_description)
        else:
            x_pct, y_pct = 0.5, 0.5
        
        # Clamp to valid range
        x_pct = max(0.0, min(1.0, x_pct))
        y_pct = max(0.0, min(1.0, y_pct))
        
        # Store pixel coords for visualization
        self._last_points = [(int(x_pct * img_width), int(y_pct * img_height))]
        
        print(f"[AttentionExtractor] Result: ({x_pct:.4f}, {y_pct:.4f})")
        return (x_pct, y_pct)
    
    def _extract_moondream(self, image: Image.Image, target_description: str) -> Tuple[float, float]:
        """Extract coordinates using moondream's native point() method."""
        import torch
        
        with torch.no_grad():
            # Use moondream's native point() method
            if hasattr(self._model, 'point'):
                print("[AttentionExtractor] Using moondream point() method")
                try:
                    result = self._model.point(image, target_description)
                    # point() returns {'points': [{'x': float, 'y': float}, ...]}
                    # x, y are already normalized to [0, 1]
                    if result and 'points' in result and len(result['points']) > 0:
                        point = result['points'][0]
                        x_pct, y_pct = point['x'], point['y']
                        print(f"[AttentionExtractor] point() returned: ({x_pct:.4f}, {y_pct:.4f})")
                        return (x_pct, y_pct)
                    else:
                        print(f"[AttentionExtractor] point() returned no points: {result}")
                except Exception as e:
                    print(f"[AttentionExtractor] point() failed: {e}")
            
            # Try detect() method as fallback
            if hasattr(self._model, 'detect'):
                print("[AttentionExtractor] Using moondream detect() method")
                try:
                    result = self._model.detect(image, target_description)
                    # detect() returns {'objects': [{'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float}, ...]}
                    # Values are normalized to [0, 1]
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
    
    def _extract_florence(self, image: Image.Image, target_description: str) -> Tuple[float, float]:
        """Extract coordinates using Florence-2 grounding."""
        import torch
        
        img_width, img_height = image.size
        
        # Florence-2 uses special task prompts
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = f"{task_prompt} {target_description}"
        
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )
        
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        
        # Parse Florence output
        parsed = self._processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(img_width, img_height)
        )
        
        print(f"[AttentionExtractor] Florence parsed: {parsed}")
        
        # Extract bounding box and get center (Florence returns pixel coords)
        if task_prompt in parsed and parsed[task_prompt].get('bboxes'):
            bboxes = parsed[task_prompt]['bboxes']
            if bboxes:
                bbox = bboxes[0]  # First match
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                # Convert to percentages
                return (x_center / img_width, y_center / img_height)
        
        # Fallback
        return (0.5, 0.5)
    
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
    
    def get_attention_heatmap(self, image_path: str, target_description: str) -> Optional[np.ndarray]:
        """Get attention heatmap if available (for visualization)."""
        self.extract_coordinates(image_path, target_description)
        return self._last_attention_grid
