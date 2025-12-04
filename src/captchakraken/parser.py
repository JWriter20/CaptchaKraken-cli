import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import torch
import base64
from PIL import Image
import io
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

# Add temp_omniparser to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OMNIPARSER_ROOT = PROJECT_ROOT / "omniparser"
if str(OMNIPARSER_ROOT) not in sys.path:
    sys.path.append(str(OMNIPARSER_ROOT))

# Now we can import from util
try:
    from util.utils import get_yolo_model, get_caption_model_processor, check_ocr_box, get_som_labeled_img
except ImportError as e:
    print(f"Warning: Could not import Omniparser utils: {e}")
    get_yolo_model = None
    get_caption_model_processor = None
    check_ocr_box = None
    get_som_labeled_img = None

@dataclass
class Component:
    id: int
    label: str
    box: List[float] # [x1, y1, x2, y2]
    type: str # 'icon' or 'text'
    confidence: float = 1.0

class CaptchaParser:
    def __init__(self, device: str = None):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"CaptchaParser using device: {self.device}")
        
        self.weights_dir = OMNIPARSER_ROOT / "weights"
        self._ensure_weights()
        
        self.config = {
            'som_model_path': str(self.weights_dir / 'icon_detect_v1_5' / 'model_v1_5.pt'),
            'caption_model_name': 'florence2',
            'caption_model_path': str(self.weights_dir / 'icon_caption_florence'),
            'BOX_TRESHOLD': 0.05
        }
        
        if get_yolo_model:
            self.som_model = get_yolo_model(model_path=self.config['som_model_path'])
            self.caption_model_processor = get_caption_model_processor(
                model_name=self.config['caption_model_name'], 
                model_name_or_path=self.config['caption_model_path'], 
                device=self.device
            )
        else:
            raise ImportError("Omniparser code not found or dependencies missing.")

    def _ensure_weights(self):
        # Only download if not present
        detect_path = self.weights_dir / "icon_detect_v1_5" / "model_v1_5.pt"
        caption_path = self.weights_dir / "icon_caption_florence"
        
        if detect_path.exists() and caption_path.exists():
            return

        print("Downloading OmniParser weights (this may take a while)...")
        if not self.weights_dir.exists():
            self.weights_dir.mkdir(parents=True)
            
        if not detect_path.exists():
            print("Downloading icon_detect weights...")
            # detect_dir = self.weights_dir / "icon_detect" # No longer needed if hf_hub_download creates it
            # detect_dir.mkdir(exist_ok=True)
            try:
                hf_hub_download(repo_id="microsoft/OmniParser", filename="icon_detect_v1_5/model_v1_5.pt", local_dir=str(self.weights_dir))
                hf_hub_download(repo_id="microsoft/OmniParser", filename="icon_detect_v1_5/model.yaml", local_dir=str(self.weights_dir))
            except Exception as e:
                print(f"Failed to download icon_detect weights: {e}")
            
        if not caption_path.exists():
            print("Downloading icon_caption_florence weights...")
            try:
                snapshot_download(repo_id="microsoft/OmniParser", allow_patterns="icon_caption_florence/*", local_dir=str(self.weights_dir))
            except Exception as e:
                print(f"Failed to download icon_caption weights: {e}")

    def parse(self, image_path: str) -> Tuple[List[Component], str]:
        """
        Parses the image and returns a list of components and the base64 encoded image with boxes drawn.
        """
        image = Image.open(image_path)
        image = image.convert('RGB')
        
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # 1. OCR
        (text, ocr_bbox), _ = check_ocr_box(
            image, 
            display_img=False, 
            output_bb_format='xyxy', 
            easyocr_args={'text_threshold': 0.8}, 
            use_paddleocr=False
        )
        
        # 2. SOM (Set-of-Mark) & Captioning
        dino_labled_img_b64, label_coordinates, parsed_content_list_elem = get_som_labeled_img(
            image, 
            self.som_model, 
            BOX_TRESHOLD=self.config['BOX_TRESHOLD'], 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=self.caption_model_processor, 
            ocr_text=text,
            use_local_semantics=True, 
            iou_threshold=0.7, 
            scale_img=False, 
            batch_size=128
        )

        components = []
        # parsed_content_list_elem is a list of dicts: {'type': 'icon'|'text', 'bbox': [x1, y1, x2, y2], 'content': str}
        # The function get_som_labeled_img sorts these such that icons with None content (if any) are at the end, 
        # but then it populates them.
        # The IDs in the annotated image correspond to the index in this list.
        
        for i, elem in enumerate(parsed_content_list_elem):
            content = elem.get('content') or "Unknown"
            bbox = elem.get('bbox') # xyxy
            comp_type = elem.get('type', 'icon')
            
            components.append(Component(
                id=i,
                label=content,
                box=bbox,
                type=comp_type
            ))
            
        return components, dino_labled_img_b64

