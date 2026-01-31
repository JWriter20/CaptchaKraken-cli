import os
import sys
import time
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

try:
    from ..attention import clusterDetections, default_predicate, complete_tracks
except (ImportError, ValueError):
    from attention import clusterDetections, default_predicate, complete_tracks

try:
    from ..image_processor import ImageProcessor
except (ImportError, ValueError):
    from image_processor import ImageProcessor

def detect(
    attention_extractor, 
    media_path: str, 
    object_class: Any, 
    max_objects: int = 24, 
    predicate=default_predicate,
    max_frames: int = 5,
    return_per_frame: bool = False
) -> Any:
    """
    Find all instances of an object class or list of classes in the image or video.
    This is the main entry point for object detection.
    
    Args:
        attention_extractor: AttentionExtractor instance
        media_path: Path to the image or video
        object_class: Description or list of descriptions of what to find
        max_objects: Maximum number of objects to return
        predicate: Optional filtering/weighting function
        max_frames: Maximum frames to process if media is a video
        return_per_frame: If True and media is video, returns detections per frame
    """
    # 1. Check for tool server fallback
    tool_server_url = os.getenv("CAPTCHA_TOOL_SERVER")
    if tool_server_url and not return_per_frame and isinstance(object_class, str):
        import requests
        try:
            # Resolve absolute path for the server
            abs_path = os.path.abspath(media_path)
            headers = {}
            api_key = os.getenv("VLLM_API_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                
            resp = requests.post(
                f"{tool_server_url}/detect",
                json={"image_path": abs_path, "text_prompt": object_class, "max_objects": max_objects},
                headers=headers,
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
                    res = det.copy()
                    res["bbox"] = [det["x_min"], det["y_min"], det["x_max"], det["y_max"]]
                    formatted.append(res)
            return formatted[:max_objects]
        except Exception as e:
            print(f"[detect] Tool server detect failed: {e}. Falling back to local.", file=sys.stderr)

    import torch
    t0 = time.time()

    # 2. Check if media is video
    is_video = any(
        media_path.lower().endswith(ext) for ext in [".mp4", ".gif", ".avi"]
    )

    detections = []
    frames_detections = [] # Used if return_per_frame is True

    if not is_video:
        # IMAGE LOGIC
        image = Image.open(media_path).convert("RGB")
        # Enhance for better detection (sharpen + color distinction)
        image = ImageProcessor.enhance_for_detection(image)
        
        width, height = image.size
        
        # Ensure models are loaded
        attention_extractor._load_sam3()
        
        # Segment using text prompt(s)
        prompts = [object_class] if isinstance(object_class, str) else object_class
        
        raw_objects = []
        for prompt in prompts:
            inputs = attention_extractor._sam3_processor(
                images=image, text=prompt, return_tensors="pt"
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
            masks = results.get("masks") # [num_objs, H, W]

            for i in range(len(scores)):
                score = float(scores[i])
                if score < 0.1:
                    continue
                    
                box = boxes[i]
                x_min, y_min, x_max, y_max = box.tolist()

                det = {
                    "x_min": x_min / width,
                    "y_min": y_min / height,
                    "x_max": x_max / width,
                    "y_max": y_max / height,
                    "score": score,
                    "prompt": prompt
                }
                if masks is not None:
                    mask = masks[i]
                    if hasattr(mask, "cpu"): mask = mask.cpu().numpy()
                    det["mask"] = mask
                
                raw_objects.append(det)

        # Merge overlapping detections
        detections = clusterDetections(raw_objects, iou_threshold=0.4, distance_threshold=0.1)
    else:
        # VIDEO LOGIC
        attention_extractor._load_sam3()
        
        # Try to load video using transformers utils, fallback to OpenCV
        try:
            from transformers.video_utils import load_video
            video_result = load_video(media_path)
            video_frames, _ = video_result
        except Exception as e:
            import cv2
            print(f"[detect] Falling back to OpenCV for video loading: {media_path}", file=sys.stderr)
            cap = cv2.VideoCapture(media_path)
            video_frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frames.append(Image.fromarray(frame_rgb))
            cap.release()

        # Handle numpy array vs list for video_frames to avoid ambiguous truth value errors
        if video_frames is None or len(video_frames) == 0:
            return []

        actual_max = min(len(video_frames), max_frames)
        video_frames = video_frames[:actual_max]

        # Enhance frames for better detection
        video_frames = [ImageProcessor.enhance_for_detection(f) for f in video_frames]

        print(f"[detect] Processing video: {media_path} ({len(video_frames)} frames)", file=sys.stderr)

        dtype = torch.bfloat16 if attention_extractor.device == "cuda" else torch.float32
        inference_session = attention_extractor._sam3_video_processor.init_video_session(
            video=video_frames,
            inference_device=attention_extractor.device,
            dtype=dtype,
        )

        # Supports single string or list of strings
        inference_session = attention_extractor._sam3_video_processor.add_text_prompt(
            inference_session=inference_session,
            text=object_class,
        )

        print(f"[detect] Propagating prompts through {actual_max} frames...", file=sys.stderr)
        tracked_paths = {}
        frame_idx = 0
        for model_outputs in attention_extractor._sam3_video_model.propagate_in_video_iterator(
            inference_session=inference_session, 
            max_frame_num_to_track=actual_max
        ):
            processed_outputs = attention_extractor._sam3_video_processor.postprocess_outputs(inference_session, model_outputs)
            
            obj_ids = processed_outputs["object_ids"].tolist()
            scores = processed_outputs["scores"].tolist()
            boxes = processed_outputs["boxes"] # [num_objs, 4] in XYXY absolute
            
            h, w = inference_session.video_height, inference_session.video_width

            current_frame_detections = []
            for i, obj_id in enumerate(obj_ids):
                if obj_id not in tracked_paths:
                    tracked_paths[obj_id] = []
                
                box = boxes[i]
                det_frame = {
                    "x_min": float(box[0]) / w,
                    "y_min": float(box[1]) / h,
                    "x_max": float(box[2]) / w,
                    "y_max": float(box[3]) / h,
                    "score": float(scores[i]),
                    "obj_id": int(obj_id),
                    "frame_idx": frame_idx,
                }
                tracked_paths[obj_id].append(det_frame)
                current_frame_detections.append(det_frame)
            
            frames_detections.append(current_frame_detections)
            frame_idx += 1

        for path in tracked_paths.values():
            if path is None or len(path) == 0: continue
            
            # Encompass all boxes in the track
            res = {
                "x_min": min(d["x_min"] for d in path),
                "y_min": min(d["y_min"] for d in path),
                "x_max": max(d["x_max"] for d in path),
                "y_max": max(d["y_max"] for d in path),
                "score": sum(d["score"] for d in path) / len(path),
            }
                
            detections.append(res)

    # 3. Format, Filter and Weight
    if is_video and return_per_frame:
        # Complete tracks (interpolation and padding)
        frames_detections = complete_tracks(frames_detections, actual_max)
        
        final_frames = []
        for frame_dets in frames_detections:
            processed_frame = []
            for det in frame_dets:
                score = det.get("score", 0.0)
                if predicate:
                    score = predicate(det)
                    if score is None:
                        continue
                
                result = det.copy()
                result["score"] = score
                result["bbox"] = [det["x_min"], det["y_min"], det["x_max"], det["y_max"]]
                processed_frame.append(result)
            processed_frame.sort(key=lambda x: x["score"], reverse=True)
            final_frames.append(processed_frame[:max_objects])
        
        # Ensure we return exactly actual_max frames even if some are empty
        while len(final_frames) < (actual_max if is_video else 0):
             final_frames.append([])
             
        return final_frames

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
    prompt_str = str(object_class) if isinstance(object_class, str) else f"{len(object_class)} prompts"
    print(f"[detect] Detection for {prompt_str} took {t1 - t0:.2f}s, found {len(final_results)} objects", file=sys.stderr)
    
    return final_results[:max_objects]

