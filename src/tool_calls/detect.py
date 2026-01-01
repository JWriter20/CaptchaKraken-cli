from typing import List, Dict, Any, Optional

def detect(attention_extractor, media_path: str, object_class: str, max_objects: int = 24) -> List[Dict[str, Any]]:
    """
    Find all instances of an object class in the image or video.
    
    Args:
        attention_extractor: AttentionExtractor instance
        media_path: Path to the image or video
        object_class: Description of what to find
        max_objects: Maximum number of objects to return
    """
    detections = attention_extractor.detect(media_path, object_class, max_objects=max_objects)
    
    # Format for the solver (package into 'bbox' key)
    formatted = []
    for det in detections:
        formatted.append({
            "bbox": [det["x_min"], det["y_min"], det["x_max"], det["y_max"]],
            "score": det.get("score", 0.0)
        })
    return formatted

