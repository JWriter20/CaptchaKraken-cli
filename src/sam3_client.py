"""Thin HTTP client for the SAM 3 segmentation service.

The service (``sam3.service``, http://127.0.0.1:8001) runs ALONGSIDE the vLLM
solver, so this needs no GPU-cycle dance. It is the only piece of the Move-drag
flow that touches the network/model; OpenCV detection (find_move_indicators /
find_movable_content) stays fully offline.

Server contract (both endpoints return JSON):
    POST /segment        fields: prompt, threshold, mask_threshold,
                                 return_masks_b64=True   + image file
    POST /segment_point  fields: x, y, box_size, return_masks_b64=True + image
    -> {num_instances, boxes:[[x1,y1,x2,y2]], scores:[..], masks_b64:[<PNG L>]}
"""

from __future__ import annotations

import base64
import io
import os
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover - requests is a project dep
    requests = None  # type: ignore

SAM3_SERVER_URL = os.environ.get("SAM3_SERVER_URL", "http://127.0.0.1:8001").rstrip("/")
_DEFAULT_SCORE_THRESHOLD = 0.25
_DEFAULT_MASK_THRESHOLD = 0.5


def server_alive(url: Optional[str] = None, timeout: float = 1.0) -> bool:
    base = (url or SAM3_SERVER_URL).rstrip("/")
    if requests is None:
        return False
    try:
        r = requests.get(base + "/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def segment_object(
    im: np.ndarray,
    *,
    point: Optional[Tuple[float, float]] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    prompt: Optional[str] = None,
    box_size: float = 0.05,
    sam3_url: Optional[str] = None,
    timeout: int = 120,
) -> dict:
    """Segment a single object out of BGR image ``im`` and return its RGBA cutout.

    Drive with exactly one of ``point`` (segment whatever is clicked) or
    ``prompt`` (segment by class name). When ``bbox`` is given the chosen mask is
    intersected with it so we cut out the object *within the card*, not a
    neighbour the model latched onto.

    Returns::
        {"rgba_png_b64": <str>,          # RGBA crop, bg transparent
         "mask_bbox": [x, y, w, h],      # cutout box in full-image px
         "score": float}

    Raises RuntimeError if the service is unreachable, ValueError if nothing is
    segmented.
    """
    base = (sam3_url or SAM3_SERVER_URL).rstrip("/")
    if not server_alive(base):
        raise RuntimeError(
            f"SAM 3 service not reachable at {base} "
            "(it normally runs as sam3.service alongside vllm)."
        )

    ok, buf = cv2.imencode(".png", im)
    if not ok:
        raise ValueError("failed to PNG-encode image for SAM 3")
    files = {"image": ("img.png", buf.tobytes(), "image/png")}

    if point is not None:
        data = {
            "x": point[0],
            "y": point[1],
            "box_size": box_size,
            "return_masks_b64": True,
        }
        resp = requests.post(base + "/segment_point", data=data, files=files, timeout=timeout)
    else:
        data = {
            "prompt": prompt or "object",
            "threshold": _DEFAULT_SCORE_THRESHOLD,
            "mask_threshold": _DEFAULT_MASK_THRESHOLD,
            "return_masks_b64": True,
        }
        resp = requests.post(base + "/segment", data=data, files=files, timeout=timeout)
    resp.raise_for_status()
    out = resp.json()

    H, W = im.shape[:2]
    masks_b64 = out.get("masks_b64") or []
    scores = out.get("scores") or [0.0] * len(masks_b64)
    if not masks_b64:
        raise ValueError("SAM 3 returned no instances")

    # Pick the mask with the largest overlap inside `bbox` (or biggest mask).
    best = None
    for m_b64, score in zip(masks_b64, scores):
        mask = _decode_mask(m_b64, (W, H))
        if bbox is not None:
            bx, by, bw, bh = bbox
            roi = np.zeros((H, W), dtype=bool)
            roi[by:by + bh, bx:bx + bw] = True
            overlap = int((mask & roi).sum())
        else:
            overlap = int(mask.sum())
        if overlap <= 0:
            continue
        if best is None or overlap > best[0]:
            best = (overlap, mask, float(score))
    if best is None:
        raise ValueError("SAM 3 mask did not overlap the requested region")

    _, mask, score = best
    rgba, mask_bbox = _cutout(im, mask)
    ok, png = cv2.imencode(".png", rgba)
    if not ok:
        raise ValueError("failed to PNG-encode RGBA cutout")
    return {
        "rgba_png_b64": base64.b64encode(png.tobytes()).decode("ascii"),
        "mask_bbox": mask_bbox,
        "score": score,
    }


def _decode_mask(m_b64: str, size: Tuple[int, int]) -> np.ndarray:
    """Decode a base64 grayscale PNG mask to a bool array of shape (H, W)."""
    raw = np.frombuffer(base64.b64decode(m_b64), dtype=np.uint8)
    m = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    W, H = size
    if m is None:
        return np.zeros((H, W), dtype=bool)
    if (m.shape[1], m.shape[0]) != (W, H):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return m > 127


def _cutout(im: np.ndarray, mask: np.ndarray):
    """Crop ``im`` to the mask bbox and return (RGBA BGRA image, [x, y, w, h])."""
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = im[y0:y1, x0:x1]
    alpha = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra, [x0, y0, x1 - x0, y1 - y0]
