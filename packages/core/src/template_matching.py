"""
Utilities for template matching–style drag puzzles.

This module exposes three primary helpers:
- extract_segment: crop a segment from a larger image using a bounding box.
- remove_background: cleanly remove the background around a cropped segment with GrabCut.
- match_template_with_mask: locate a cleaned segment inside a base image using masked template matching.

A small helper (find_draggable_candidates) is included to approximate the
initial bounding box detection that would normally come from the model
moondream's detect() API. It uses simple HSV filtering to find saturated,
square-ish regions (the typical draggable tiles in hCaptcha puzzles).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


@dataclass
class MatchResult:
    """Result of a template match."""

    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    center: Tuple[float, float]
    confidence: float
    method: int


def _load_image(image: Union[str, Path, np.ndarray], with_alpha: bool = False) -> np.ndarray:
    """Load an image from disk unless an array is already provided."""
    if isinstance(image, np.ndarray):
        return image

    flags = cv2.IMREAD_UNCHANGED if with_alpha else cv2.IMREAD_COLOR
    img = cv2.imread(str(image), flags)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image}")
    return img


def extract_segment(
    image_path: Union[str, Path, np.ndarray],
    bbox: BBox,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Crop a segment from the source image.

    Args:
        image_path: Path to the source image or an already-loaded array.
        bbox: (x1, y1, x2, y2) pixel coordinates of the segment to crop.
        output_path: Optional path to save the cropped image.

    Returns:
        Cropped image array in BGR order.
    """
    img = _load_image(image_path)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1] - 1), min(y2, img.shape[0] - 1)

    cropped = img[y1 : y2 + 1, x1 : x2 + 1].copy()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cropped)

    return cropped


def remove_background(
    image: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    rect_margin: int = 5,
    iterations: int = 5,
) -> np.ndarray:
    """
    Remove the background from a cropped segment using GrabCut.

    Args:
        image: Path or image array containing the cropped segment.
        output_path: Optional path to save the RGBA result.
        rect_margin: How many pixels of margin to leave around the border
            for the initial GrabCut rectangle.
        iterations: Number of GrabCut refinement iterations.

    Returns:
        RGBA image with an alpha channel (0 = transparent, 255 = opaque).
    """
    bgr = _load_image(image)
    h, w = bgr.shape[:2]

    if h < 2 or w < 2:
        raise ValueError("Segment too small to process")

    # Prepare mask and background/foreground models
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Rectangle that likely contains the foreground
    rect = (
        rect_margin,
        rect_margin,
        max(1, w - 2 * rect_margin),
        max(1, h - 2 * rect_margin),
    )

    # Run GrabCut, fallback to simple opaque mask if it fails
    try:
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    except Exception:
        mask = np.ones((h, w), np.uint8)

    alpha = (mask * 255).astype("uint8")
    rgba = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), rgba)

    return rgba


def _apply_exclusions(base: np.ndarray, exclude_regions: Optional[Sequence[BBox]]) -> np.ndarray:
    """Zero out regions that should not be considered during matching."""
    if not exclude_regions:
        return base

    masked = base.copy()
    for x1, y1, x2, y2 in exclude_regions:
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(int(x2), masked.shape[1] - 1), min(int(y2), masked.shape[0] - 1)
        masked[y1 : y2 + 1, x1 : x2 + 1] = 0
    return masked


def match_template_with_mask(
    base_image: Union[str, Path, np.ndarray],
    template_image: Union[str, Path, np.ndarray],
    method: int = cv2.TM_CCORR_NORMED,
    exclude_regions: Optional[Sequence[BBox]] = None,
) -> MatchResult:
    """
    Find where a (possibly transparent) template best fits inside a base image.

    Args:
        base_image: Path or array for the base image.
        template_image: Path or array for the template. May contain an alpha channel.
        method: OpenCV template matching method (default: TM_CCORR_NORMED).
        exclude_regions: Optional regions in the base image to ignore (e.g., the
            tray where the draggable piece currently sits).

    Returns:
        MatchResult with the best location and confidence score.
    """
    base = _load_image(base_image)
    templ = _load_image(template_image, with_alpha=True)

    # Split template and mask if present
    if templ.shape[2] == 4:
        mask = templ[:, :, 3]
        templ_bgr = templ[:, :, :3]
    else:
        mask = None
        templ_bgr = templ

    base_proc = _apply_exclusions(base, exclude_regions).astype(np.float32)
    templ_proc = templ_bgr.astype(np.float32)

    use_mask = mask is not None and np.count_nonzero(mask) > 0
    mask_proc = (mask / 255.0).astype(np.float32) if use_mask else None

    result = None

    if use_mask:
        try:
            result = cv2.matchTemplate(base_proc, templ_proc, method, mask=mask_proc)
            # Guard against NaNs/Infs produced by degenerate masks
            result = np.nan_to_num(result, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
            if not np.isfinite(result).any():
                result = None
        except Exception:
            result = None

    # Fallback to unmasked matching if masked variant failed
    if result is None:
        result = cv2.matchTemplate(base_proc, templ_proc, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # For normalized correlation methods, higher is better.
    top_left = max_loc if method not in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED) else min_loc
    h, w = templ_bgr.shape[:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    center = (top_left[0] + w / 2.0, top_left[1] + h / 2.0)
    confidence = max_val if method not in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED) else -min_val

    return MatchResult(
        top_left=top_left,
        bottom_right=bottom_right,
        center=center,
        confidence=float(confidence),
        method=method,
    )


def find_draggable_candidates(
    image_path: Union[str, Path, np.ndarray],
    s_thresh: int = 80,
    v_thresh: int = 70,
    area_range: Tuple[int, int] = (3000, 120_000),
    aspect_range: Tuple[float, float] = (0.6, 1.4),
    min_y_ratio: float = 0.0,
) -> List[dict]:
    """
    Heuristically find square, high-saturation regions that often correspond
    to draggable tiles. This stands in for the model-based detection described
    in AlgoImprovements.txt and is used by tests as a deterministic proxy.

    Returns:
        A list of dictionaries sorted by area descending with keys:
        {"bbox": (x1, y1, x2, y2), "area": int, "aspect": float}
    """
    img = _load_image(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        (0, s_thresh, v_thresh),
        (180, 255, 255),
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    y_min = int(h * min_y_ratio)

    candidates: List[dict] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < area_range[0] or area > area_range[1]:
            continue
        if y < y_min:
            continue
        aspect = bw / float(bh) if bh else 0.0
        if not (aspect_range[0] <= aspect <= aspect_range[1]):
            continue
        candidates.append({"bbox": (x, y, x + bw - 1, y + bh - 1), "area": area, "aspect": aspect})

    candidates.sort(key=lambda c: c["area"], reverse=True)
    return candidates


def solve_drag_pipeline(
    base_image_path: Union[str, Path],
    segment_bbox: BBox,
    output_dir: Optional[Union[str, Path]] = None,
) -> MatchResult:
    """
    Convenience wrapper that runs:
    1) crop -> 2) background removal -> 3) masked template matching.
    """
    output_dir = Path(output_dir) if output_dir else None

    cropped = extract_segment(
        base_image_path,
        segment_bbox,
        output_path=(output_dir / "segment.png") if output_dir else None,
    )
    cleaned = remove_background(
        cropped,
        output_path=(output_dir / "segment_nobg.png") if output_dir else None,
    )

    # Use the cleaned template for matching; exclude the source tray area.
    result = match_template_with_mask(
        base_image_path,
        cleaned,
        exclude_regions=[segment_bbox],
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        vis = _load_image(base_image_path).copy()
        tl = tuple(map(int, result.top_left))
        br = tuple(map(int, result.bottom_right))
        cv2.rectangle(vis, tl, br, (0, 255, 0), 2)
        cv2.imwrite(str(output_dir / "match_result.png"), vis)

    return result


__all__ = [
    "BBox",
    "MatchResult",
    "extract_segment",
    "remove_background",
    "match_template_with_mask",
    "find_draggable_candidates",
    "solve_drag_pipeline",
]

