from typing import List, Optional, Tuple

import cv2
import numpy as np


def to_greyscale(image_path: str, output_path: str) -> None:
    """
    Convert an image to greyscale and save it.

    Args:
        image_path: Path to the input image
        output_path: Path where the greyscale image will be saved
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, gray)


def to_edge_outline(image_path: str, output_path: str, low_threshold: int = 15, high_threshold: int = 60) -> None:
    """
    Detect and isolate edges in an image using Canny edge detection.

    Args:
        image_path: Path to the input image
        output_path: Path where the edge-detected image will be saved
        low_threshold: Lower threshold for Canny edge detection (default: 50)
        high_threshold: Upper threshold for Canny edge detection (default: 150)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Convert to greyscale first (required for Canny)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    cv2.imwrite(output_path, edges)


def get_grid_bounding_boxes(image_path: str) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Detects a grid of images separated by white spacing (e.g., 3x3 or 4x4 captchas).
    Returns a list of bounding boxes (x1, y1, x2, y2) for the grid cells.
    Returns None if no grid structure is found.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to find white lines (spacing).
    # Using a high threshold (240) to detect white or near-white spacing.
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    h, w = binary.shape

    # Horizontal projection to find rows
    row_sums = np.sum(binary, axis=1) / 255
    row_ratio = row_sums / w

    # Find potential horizontal lines (ratio > 0.85)
    # This allows for some noise or objects crossing the line slightly
    threshold_ratio = 0.85
    row_indices = np.where(row_ratio > threshold_ratio)[0]

    def group_indices(indices):
        if len(indices) == 0:
            return []
        groups = []
        current_group = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                current_group.append(indices[i])
            else:
                groups.append(int(np.mean(current_group)))
                current_group = [indices[i]]
        groups.append(int(np.mean(current_group)))
        return groups

    h_lines = group_indices(row_indices)

    # We need at least 3 lines to form 2 rows (top, mid, bot)
    if len(h_lines) < 3:
        return None

    possible_grids = []

    # Search for evenly spaced sequences of horizontal lines
    for i in range(len(h_lines)):
        for j in range(i + 2, len(h_lines) + 1):
            subset = h_lines[i:j]
            if len(subset) < 3:
                continue

            # Check spacing consistency
            spacings = [subset[k] - subset[k - 1] for k in range(1, len(subset))]
            avg_spacing = np.mean(spacings)
            std_spacing = np.std(spacings)

            # Allow variance
            if std_spacing < max(5, avg_spacing * 0.1):
                n_rows = len(subset) - 1
                # We are interested in 3 or 4 rows typically
                if n_rows in [3, 4]:
                    possible_grids.append((subset, n_rows))

    valid_grids = []

    for lines, n_rows in possible_grids:
        y_min = lines[0]
        y_max = lines[-1]

        # Crop to the vertical extent of the potential grid
        crop = binary[y_min:y_max, :]
        crop_h, crop_w = crop.shape

        # Vertical projection on the crop
        col_sums = np.sum(crop, axis=0) / 255
        col_ratio = col_sums / crop_h

        v_indices = np.where(col_ratio > threshold_ratio)[0]
        v_lines_cand = group_indices(v_indices)

        # Filter internal vertical lines (exclude near-borders)
        v_lines_internal = [x for x in v_lines_cand if x > crop_w * 0.05 and x < crop_w * 0.95]

        # We expect n_cols columns.
        # n_cols = len(v_lines_internal) + 1 (assuming borders are implicit or handled)
        n_cols = len(v_lines_internal) + 1

        # Check for standard grid shapes
        # 3x3, 4x4 are most common. 3x4 or 4x3 are possible.
        is_valid = False
        if n_rows == 3 and n_cols == 3:
            is_valid = True
        elif n_rows == 4 and n_cols == 4:
            is_valid = True
        elif n_rows == 3 and n_cols == 4:
            is_valid = True
        elif n_rows == 4 and n_cols == 3:
            is_valid = True

        if is_valid:
            valid_grids.append(
                {
                    "h_lines": lines,
                    "v_lines_internal": v_lines_internal,
                    "rows": n_rows,
                    "cols": n_cols,
                    "area": (lines[-1] - lines[0]) * w,  # Approximate area
                }
            )

    if not valid_grids:
        return None

    # Pick the largest grid (area) or most cells
    # Heuristic: max(rows * cols), then max area
    best_grid = max(valid_grids, key=lambda x: (x["rows"] * x["cols"], x["area"]))

    # Generate boxes
    h_lines_found = best_grid["h_lines"]
    v_lines_internal = best_grid["v_lines_internal"]
    rows = best_grid["rows"]
    cols = best_grid["cols"]

    # Refine horizontal lines using internal spacing
    # This helps avoid issues where the detected top/bottom lines are slightly off
    # or include headers/footers (e.g. extending too far).
    if rows >= 3:
        # We have at least 2 internal separators (lines[1]...lines[-2]) for 3 rows (lines 0..3)
        # Calculate average height from internal segments only
        internal_h_lines = h_lines_found[1:-1]
        if len(internal_h_lines) >= 2:
            spacings_h = [internal_h_lines[i] - internal_h_lines[i - 1] for i in range(1, len(internal_h_lines))]
            avg_h_spacing = np.mean(spacings_h)
        else:
            # Fallback for 3 rows (lines 0, 1, 2, 3): internal are 1, 2. Spacing is just (2-1).
            avg_h_spacing = internal_h_lines[-1] - internal_h_lines[0]

        # Reconstruct top and bottom based on internal anchors
        # Top = first_internal - spacing
        # Bottom = last_internal + spacing
        new_top = int(internal_h_lines[0] - avg_h_spacing)
        new_bottom = int(internal_h_lines[-1] + avg_h_spacing)

        # Update boundaries (keep internal lines as is)
        h_lines_found = [new_top] + list(internal_h_lines) + [new_bottom]

    # Determine vertical boundaries
    # We can estimate the full vertical lines including borders
    if len(v_lines_internal) > 1:
        avg_v_spacing = np.mean(
            [v_lines_internal[k] - v_lines_internal[k - 1] for k in range(1, len(v_lines_internal))]
        )
    else:
        avg_v_spacing = w / cols

    # Try to find existing border lines in v_lines_cand (not computed here, need to re-scan or just estimate)
    # Estimate borders based on spacing
    # left ~ first_internal - spacing
    # right ~ last_internal + spacing

    left_border = max(0, int(v_lines_internal[0] - avg_v_spacing))
    right_border = min(w, int(v_lines_internal[-1] + avg_v_spacing))

    # Better: check if there's a line at left_border +/- tolerance
    # But estimation is robust enough for bounding boxes usually

    # Construct x-coordinates
    # We have v_lines_internal. We add left/right borders.
    # Note: If v_lines_internal is [x1, x2], we want [x0, x1, x2, x3]
    # Check if we should insert the borders

    x_coords = [left_border] + v_lines_internal + [right_border]

    # Verify count
    if len(x_coords) != cols + 1:
        # If estimation failed (e.g. 4 cols but we only found 2 internal lines?), fallback to equal division
        x_step = (right_border - left_border) / cols
        x_coords = [int(left_border + k * x_step) for k in range(cols + 1)]

    boxes = []
    for r in range(rows):
        for c in range(cols):
            y1 = int(h_lines_found[r])
            y2 = int(h_lines_found[r + 1])
            x1 = int(x_coords[c])
            x2 = int(x_coords[c + 1])

            # Optional: Add margin? The lines are the separators.
            # Usually we want the content inside.
            # The current coords include half the separator line width if we split exactly?
            # Actually our 'lines' are center of separators.
            # So boxes are slightly overlapping the separators.
            # It's usually fine, or we can shrink by 1-2 pixels.

            boxes.append((x1, y1, x2, y2))

    return boxes
