import os
import sys
import time
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

try:
    from ..attention import clusterDetections, default_predicate
except (ImportError, ValueError):
    from attention import clusterDetections, default_predicate

def detect(
    attention_extractor, 
    media_path: str, 
    object_class: str, 
    max_objects: int = 24, 
    predicate=default_predicate,
    max_frames: int = 5
) -> List[Dict[str, Any]]:
    """
    Find all instances of an object class in the image or video.
    This is the main entry point for object detection.
    
    Args:
        attention_extractor: AttentionExtractor instance
        media_path: Path to the image or video
        object_class: Description of what to find
        max_objects: Maximum number of objects to return
        predicate: Optional filtering/weighting function
        max_frames: Maximum frames to process if media is a video
    """
    # 1. Check for tool server fallback
    tool_server_url = os.getenv("CAPTCHA_TOOL_SERVER")
    if tool_server_url:
        import requests
        try:
            # Resolve absolute path for the server
            abs_path = os.path.abspath(media_path)
            resp = requests.post(
                f"{tool_server_url}/detect",
                json={"image_path": abs_path, "text_prompt": object_class, "max_objects": max_objects},
                timeout=300
            )
            resp.raise_for_status()
            objects = resp.json().get("objects", [])
            
            # Format server results
            formatted = []
            for det in objects:
                # Server might already return formatted bboxes or raw detections
                if "bbox" in det:
                    formatted.append(det)
                else:
                    formatted.append({
                        "bbox": [det["x_min"], det["y_min"], det["x_max"], det["y_max"]],
                        "score": det.get("score", 0.0)
                    })
            return formatted[:max_objects]
        except Exception as e:
            print(f"[detect] Tool server detect failed: {e}. Falling back to local.", file=sys.stderr)

    import torch
    t0 = time.time()

    # 2. Check if media is video
    is_video = any(
        media_path.lower().endswith(ext) for ext in [".mp4", ".webm", ".gif", ".avi"]
    )

    detections = []

    if not is_video:
        # IMAGE LOGIC
        image = Image.open(media_path).convert("RGB")
        width, height = image.size
        
        # Ensure models are loaded
        attention_extractor._load_sam3()
        
        # Segment using text prompt
        inputs = attention_extractor._sam3_processor(
            images=image, text=object_class, return_tensors="pt"
        ).to(attention_extractor.device)
        
        # Ensure floating point inputs match model dtype
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                inputs[k] = v.to(attention_extractor._sam3_model.dtype)

        with torch.no_grad():
            outputs = attention_extractor._sam3_model(**inputs)

        # Post-process results
        results = attention_extractor._sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.1,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        boxes = results["boxes"] # [num_objs, 4] in XYXY absolute
        scores = results["scores"] # [num_objs]

        raw_objects = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score < 0.1:
                continue
                
            box = boxes[i]
            x_min, y_min, x_max, y_max = box.tolist()

            raw_objects.append({
                "x_min": x_min / width,
                "y_min": y_min / height,
                "x_max": x_max / width,
                "y_max": y_max / height,
                "score": score,
            })

        # Merge overlapping detections
        detections = clusterDetections(raw_objects, iou_threshold=0.4, distance_threshold=0.1)
    else:
        # VIDEO LOGIC
        attention_extractor._load_sam3()
        from transformers.video_utils import load_video
        
        video_frames, _ = load_video(media_path)
        actual_max = min(len(video_frames), max_frames)
        video_frames = video_frames[:actual_max]

        print(f"[detect] Processing video: {media_path} ({len(video_frames)} frames)", file=sys.stderr)

        dtype = torch.bfloat16 if attention_extractor.device == "cuda" else torch.float32
        inference_session = attention_extractor._sam3_video_processor.init_video_session(
            video=video_frames,
            inference_device=attention_extractor.device,
            dtype=dtype,
        )

        inference_session = attention_extractor._sam3_video_processor.add_text_prompt(
            inference_session=inference_session,
            text=object_class,
        )

        tracked_paths = {}
        for model_outputs in attention_extractor._sam3_video_model.propagate_in_video_iterator(
            inference_session=inference_session, 
            max_frame_num_to_track=actual_max
        ):
            processed_outputs = attention_extractor._sam3_video_processor.postprocess_outputs(inference_session, model_outputs)
            
            obj_ids = processed_outputs["object_ids"].tolist()
            scores = processed_outputs["scores"].tolist()
            boxes = processed_outputs["boxes"] # [num_objs, 4] in XYXY absolute
            
            h, w = inference_session.video_height, inference_session.video_width

            for i, obj_id in enumerate(obj_ids):
                if obj_id not in tracked_paths:
                    tracked_paths[obj_id] = []
                
                box = boxes[i]
                tracked_paths[obj_id].append({
                    "x_min": float(box[0]) / w,
                    "y_min": float(box[1]) / h,
                    "x_max": float(box[2]) / w,
                    "y_max": float(box[3]) / h,
                    "score": float(scores[i]),
                })

        for path in tracked_paths.values():
            if not path: continue
            detections.append({
                "x_min": min(d["x_min"] for d in path),
                "y_min": min(d["y_min"] for d in path),
                "x_max": max(d["x_max"] for d in path),
                "y_max": max(d["y_max"] for d in path),
                "score": sum(d["score"] for d in path) / len(path),
            })

    # 3. Format, Filter and Weight
    final_results = []
    for det in detections:
        score = det.get("score", 0.0)
        if predicate:
            score = predicate(det)
            if score is None:
                continue
        
        # Keep all original keys plus add bbox for convenience
        result = det.copy()
        result["score"] = score
        result["bbox"] = [det["x_min"], det["y_min"], det["x_max"], det["y_max"]]
        final_results.append(result)
    
    final_results.sort(key=lambda x: x["score"], reverse=True)
    
    t1 = time.time()
    print(f"[detect] Detection for '{object_class}' took {t1 - t0:.2f}s, found {len(final_results)} objects", file=sys.stderr)
    
    return final_results[:max_objects]

