"""
Tests for the OpenCV-based template matching utilities.

These tests simulate the first dragging goal for the template-matching
drag puzzles by:
- estimating a draggable bounding box (proxy for moondream detect())
- cropping the segment
- removing its background
- matching it back into the base image
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.template_matching import (
    extract_segment,
    find_draggable_candidates,
    match_template_with_mask,
    remove_background,
    solve_drag_pipeline,
)


IMAGES_DIR = Path(__file__).resolve().parent.parent / "captchaimages"


@pytest.mark.parametrize("filename", ["hcaptchaDragImage1.png", "hcaptchaDragImage2.png"])
def test_drag_pipeline_end_to_end(tmp_path, filename):
    """
    Run the full pipeline on the provided drag captcha images.
    """
    base_path = IMAGES_DIR / filename
    assert base_path.exists(), f"Missing test image: {base_path}"

    # Step 1: approximate the draggable bbox (stand-in for moondream detect()).
    candidates = find_draggable_candidates(base_path, min_y_ratio=0.15)
    assert candidates, f"No draggable candidates found for {filename}"
    bbox = candidates[0]["bbox"]

    # Step 2: crop and remove background.
    cropped_path = tmp_path / f"{filename}_segment.png"
    rgba_path = tmp_path / f"{filename}_segment_rgba.png"
    cropped = extract_segment(base_path, bbox, output_path=cropped_path)
    cleaned = remove_background(cropped, output_path=rgba_path)

    assert cleaned.shape[2] == 4, "Expected RGBA output after background removal"
    alpha = cleaned[:, :, 3]
    assert alpha.max() == 255
    assert np.count_nonzero(alpha) > 0
    assert np.count_nonzero(alpha == 0) > 0  # ensure some background was removed

    # Step 3: match the cleaned segment back into the base (ignoring tray area).
    result = match_template_with_mask(base_path, cleaned, exclude_regions=[bbox])
    base_img = cv2.imread(str(base_path))
    bh, bw = base_img.shape[:2]

    assert np.isfinite(result.confidence)
    assert 0 <= result.top_left[0] < bw
    assert 0 <= result.top_left[1] < bh
    assert result.bottom_right[0] <= bw
    assert result.bottom_right[1] <= bh


@pytest.mark.parametrize("filename", ["hcaptchaDragImage1.png", "hcaptchaDragImage2.png"])
def test_solve_drag_pipeline_helper(tmp_path, filename):
    """
    Ensure the convenience wrapper returns a MatchResult and saves artifacts when requested.
    """
    base_path = IMAGES_DIR / filename
    candidates = find_draggable_candidates(base_path, min_y_ratio=0.15)
    assert candidates, f"No draggable candidates found for {filename}"
    bbox = candidates[0]["bbox"]

    output_dir = tmp_path / "artifacts"
    result = solve_drag_pipeline(base_path, bbox, output_dir=output_dir)

    assert output_dir.exists()
    assert (output_dir / "segment.png").exists()
    assert (output_dir / "segment_nobg.png").exists()
    assert (output_dir / "match_result.png").exists()

    base_img = cv2.imread(str(base_path))
    bh, bw = base_img.shape[:2]
    assert 0 <= result.center[0] <= bw
    assert 0 <= result.center[1] <= bh
    assert np.isfinite(result.confidence)

