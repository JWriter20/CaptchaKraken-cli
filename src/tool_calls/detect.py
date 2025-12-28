from typing import List, Dict, Any, Optional

def detect(attention_extractor, image_path: str, object_class: str, max_objects: int = 24) -> List[Dict[str, Any]]:
    """
    Find all instances of an object class in the image.
    
    Args:
        attention_extractor: AttentionExtractor instance
        image_path: Path to the image
        object_class: Description of what to find
        max_objects: Maximum number of objects to return
    """
    detections = attention_extractor.detect(image_path, object_class, max_objects=max_objects)
    return detections

