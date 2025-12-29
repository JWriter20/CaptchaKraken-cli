import cv2
import numpy as np
import os
import tempfile
from typing import List, Tuple, Optional
from ..overlay import add_overlays_to_image
from ..image_processor import ImageProcessor

def find_grid(image_path: str, debug_manager=None, slant_to_try: Optional[float] = None) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Main entry point for grid detection.
    Detects a 3x3 or 4x4 grid by looking for grey/white separator lines.
    Supports slanted grids (common in some reCAPTCHA variants).
    """
    img = cv2.imread(image_path)
    if img is None: return None
    h, w = img.shape[:2]
    
    # Use LAB color space for more consistent color distance calculations
    # especially for grey/white line detection.
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Normalize LAB to standard scales (L: 0-100, a,b: -128-127) for use with ImageProcessor
    img_lab[:, :, 0] *= (100.0 / 255.0)
    img_lab[:, :, 1] -= 128.0
    img_lab[:, :, 2] -= 128.0
    
    def find_all_candidates(lines, total_dim):
        """
        Takes a list of line positions and finds sets of lines that could form a grid.
        For a 3x3 grid, we need 2 internal lines.
        For a 4x4 grid, we need 3 internal lines.
        """
        margin = 0.02
        # Filter out lines too close to the edge
        internal_lines = [l for l in lines if l[0] > total_dim * margin and l[0] < total_dim * (1.0 - margin)]
        
        candidates = {2: [], 3: []}
        for i in range(len(internal_lines)):
            p1, l1, c1 = internal_lines[i]
            for j in range(i + 1, len(internal_lines)):
                p2, l2, c2 = internal_lines[j]
                d = p2 - p1
                # Check for 2-line candidates (for 3x3 grid)
                if 40 < d < 600:
                    # Calculate how centered the pair is
                    center_offset = abs((p1 + p2) / 2.0 - total_dim / 2.0) / total_dim
                    # Scale penalty: a 3x3 grid cell should be roughly 1/3 of the image
                    scale_err = abs(d - total_dim / 3.0) / total_dim
                    score = center_offset * 1000 + scale_err * 500
                    candidates[2].append(([p1, p2], score, d, min(l1, l2), [c1, c2]))
                
                # Check for 3-line candidates (for 4x4 grid)
                for k in range(j + 1, len(internal_lines)):
                    p3, l3, c3 = internal_lines[k]
                    d1, d2 = p2 - p1, p3 - p2
                    if 40 < d1 < 600 and 40 < d2 < 600:
                        # Ensure lines are roughly equally spaced
                        rel_diff = abs(d1 - d2) / max(d1, d2)
                        if rel_diff < 0.35:
                            center_offset = abs((p1 + p3) / 2.0 - total_dim / 2.0) / total_dim
                            # Scale penalty: a 4x4 grid cell should be roughly 1/4 of the image
                            scale_err = abs(((d1 + d2) / 2.0) - total_dim / 4.0) / total_dim
                            score = rel_diff * 100 + center_offset * 1000 + scale_err * 500
                            candidates[3].append(([p1, p2, p3], score, (d1 + d2) / 2.0, min(l1, l2, l3), [c1, c2, c3]))

        # Sort candidates by score
        for k in candidates:
            candidates[k].sort(key=lambda x: x[1])
        return candidates

    def run_detection(slant, threshold=3.0, color_thr=6.0):
        """
        Attempts to detect a grid with a specific slant factor.
        """
        # Get candidate horizontal and vertical lines
        h_lines, h_raw = _get_candidate_lines(img_lab, axis=1, slant=slant, threshold=threshold)
        v_lines, v_raw = _get_candidate_lines(img_lab, axis=0, slant=-slant, threshold=threshold)
        
        if not h_lines or not v_lines: return None, float('inf')

        # Find sets of lines that could form a grid
        h_cand = find_all_candidates(h_lines, h)
        v_cand = find_all_candidates(v_lines, w)
        
        best_run_grid = None
        min_run_score = float('inf')
        
        # Try both 3x3 and 4x4 configurations
        for size in [3, 4]:
            n = size - 1 # number of internal lines needed
            if h_cand[n] and v_cand[n]:
                # Cross-reference horizontal and vertical candidates
                for hs, hs_sc, hd, h_len, h_colors in h_cand[n][:20]:
                    for vs, vs_sc, vd, v_len, v_colors in v_cand[n][:20]:
                        # Consistency check: all separator lines should have similar colors
                        all_used_colors = h_colors + v_colors
                        avg_c = np.mean(all_used_colors, axis=0)
                        if all(np.sqrt(np.sum((c - avg_c)**2)) < color_thr for c in all_used_colors):
                            grid_boxes = _generate_slanted_grid(size, hs, vs, hd, vd, h, w, slant)
                            if grid_boxes:
                                # Penalize non-square grids and extreme slants
                                # Also prefer 4x4 grids over 3x3 if both are found
                                s_diff = abs(hd - vd) / max(hd, vd)
                                size_bonus = (4 - size) * 100 # Penalty for size 3
                                score = hs_sc + vs_sc + s_diff * 1000 + abs(slant) * 500 + size_bonus
                                if score < min_run_score:
                                    min_run_score, best_run_grid = score, grid_boxes
        
        # Debugging: Save image showing candidate lines
        if debug_manager and getattr(debug_manager, 'enabled', False):
            dbg_img = img.copy()
            # Draw all candidate lines found
            for pos, _, _ in h_raw:
                y1 = int(pos + slant * (0 - w//2))
                y2 = int(pos + slant * (w - 1 - w//2))
                cv2.line(dbg_img, (0, y1), (w-1, y2), (255, 0, 0), 1)
            for pos, _, _ in v_raw:
                x1 = int(pos - slant * (0 - h//2))
                x2 = int(pos - slant * (h - 1 - h//2))
                cv2.line(dbg_img, (x1, 0), (x2, h-1), (0, 255, 0), 1)
            
            image_basename = os.path.basename(image_path)
            slant_str = f"{slant:.3f}".replace('.', '_')
            debug_path = os.path.join(str(getattr(debug_manager, 'base_dir', ".")), f"lines_slant_{slant_str}_{image_basename}")
            cv2.imwrite(debug_path, dbg_img)

        return best_run_grid, min_run_score

    slant_val = slant_to_try if slant_to_try is not None else 0.0
    # Default logic: try no slant first, then others
    best_grid, min_score = run_detection(slant_val, threshold=1.0, color_thr=10.0)
    best_slant = slant_val
    
    # If no grid found or result is poor, try various slant angles
    if not best_grid or min_score > 100:
        for slant in [0.015, -0.015]:
            grid, score = run_detection(slant, threshold=3.0, color_thr=10.0)
            if grid and score < min_score:
                min_score, best_grid = score, grid
                best_slant = slant

    if not best_grid: return None
    
    # Final debug output: Save image with detected grid boxes
    if debug_manager and getattr(debug_manager, 'enabled', False):
        image_basename = os.path.basename(image_path)
        # Use the best slant found in filename
        slant_str = f"{best_slant:.3f}".replace('.', '_')
        debug_path = os.path.join(str(getattr(debug_manager, 'base_dir', ".")), f"grid_final_slant_{slant_str}_{image_basename}")
        get_numbered_grid_overlay(image_path, best_grid, output_path=debug_path)
        
    return best_grid

def _generate_slanted_grid(size, hs, vs, hd, vd, h, w, slant):
    """
    Calculates the bounding boxes for each cell in the grid, 
    accounting for the slant factor.
    """
    mid_x, mid_y = w / 2, h / 2
    # Extrapolate grid boundaries from detected internal lines
    if size == 4:
        # Use all three detected lines as anchors for the grid
        y_bounds = [hs[0] - hd, hs[0], hs[1], hs[2], hs[2] + hd]
        x_bounds = [vs[0] - vd, vs[0], vs[1], vs[2], vs[2] + vd]
    else:
        # Use both detected lines as anchors
        y_bounds = [hs[0] - hd, hs[0], hs[1], hs[1] + hd]
        x_bounds = [vs[0] - vd, vs[0], vs[1], vs[1] + vd]

    grid_boxes = []
    s2 = slant * slant
    denom = 1 + s2
    # For each cell in the grid, find the corners and then the bounding box
    for i in range(len(y_bounds)-1):
        for j in range(len(x_bounds)-1):
            corners = []
            for y_i in [y_bounds[i], y_bounds[i+1]]:
                for x_j in [x_bounds[j], x_bounds[j+1]]:
                    # Apply inverse slant transformation to find actual pixel coordinates
                    cy = (y_i + slant * (x_j - mid_x) + s2 * mid_y) / denom
                    cx = x_j - slant * (cy - mid_y)
                    corners.append((cx, cy))
            pts = np.array(corners, dtype=np.float32)
            x1, y1 = np.min(pts, axis=0)
            x2, y2 = np.max(pts, axis=0)
            # Clip to image boundaries
            x1_c, y1_c = int(max(0, min(w, x1))), int(max(0, min(h, y1)))
            x2_c, y2_c = int(max(0, min(w, x2))), int(max(0, min(h, y2)))
            if x2_c > x1_c and y2_c > y1_c:
                grid_boxes.append((x1_c, y1_c, x2_c, y2_c))
    
    # Only return if we found all expected boxes
    return grid_boxes if len(grid_boxes) == size*size else None

def _get_candidate_lines(img_lab, axis, slant, threshold):
    """
    Scans the image for lines that match the expected grid separator color (grey/white).
    Optimized using NumPy for performance.
    axis=1: Horizontal lines
    axis=0: Vertical lines
    """
    h, w = img_lab.shape[:2]
    mid_x, mid_y = w / 2, h / 2
    
    # Perceptual neutrality check: calculate chrominance (distance from neutral in LAB)
    A = img_lab[:, :, 1]
    B = img_lab[:, :, 2]
    chroma = np.sqrt(A**2 + B**2)
    
    # Grid lines are very bright (L > 50) and almost perfectly neutral (low chroma)
    # Lowering chroma threshold to 4.0 to strictly exclude sky blue and other tints
    grey_mask = (img_lab[:, :, 0] > 80) & (chroma < 5.5)
    grey_mask_u8 = grey_mask.astype(np.uint8) * 255

    candidates = []
    
    if axis == 1: # Horizontal-ish
        # Intelligent selective check: check center for potential lines
        center_x = w // 2
        cx_start, cx_end = max(0, center_x - 1), min(w, center_x + 2)
        # Note: we use the grey_mask directly here, but we will scan along slanted paths
        possible_y0s = np.where(np.any(grey_mask[:, cx_start:cx_end] > 0, axis=1))[0]
        
        if len(possible_y0s) > 0:
            central_start, central_end = int(w * 0.15), int(w * 0.85)
            central_width = central_end - central_start
            x_range = np.arange(w)
            
            for y0 in possible_y0s:
                # Slanted scanning: calculate the Y coordinates along a slanted line starting at y0
                # y_src = y0 + slant * (x - mid_x)
                y_indices = np.round(y0 + slant * (x_range - mid_x)).astype(np.int32)
                
                # Check if the central 70% is within image bounds
                central_y = y_indices[central_start:central_end]
                if np.any(central_y < 0) or np.any(central_y >= h):
                    continue
                
                # Extract the central slanted segment from the mask
                central_mask = grey_mask[central_y, x_range[central_start:central_end]]
                
                # The central segment must be mostly grey (allowing a 10% margin for noise/artifacts)
                if np.sum(central_mask) < central_width * 0.9:
                    continue
                
                # Color consistency check for the central 70%
                segment_colors = img_lab[central_y, x_range[central_start:central_end]]
                
                # Consistency check: Std dev of colors should be low
                std_dev = np.std(segment_colors, axis=0)
                if np.all(std_dev < threshold):
                    avg_color = np.mean(segment_colors, axis=0)
                    candidates.append((float(y0), float(central_width), avg_color))
    else: # Vertical-ish
        # Center check: use a 3-pixel window for robustness
        center_y = h // 2
        cy_start, cy_end = max(0, center_y - 1), min(h, center_y + 2)
        possible_x0s = np.where(np.any(grey_mask[cy_start:cy_end, :], axis=0))[0]
        
        if len(possible_x0s) > 0:
            central_start, central_end = int(h * 0.25), int(h * 0.75) # Central 50%
            central_height = central_end - central_start
            y_range = np.arange(h)
            
            for x0 in possible_x0s:
                # Slanted scanning for vertical lines
                # x_src = x0 + slant * (y - mid_y)
                x_indices = np.round(x0 + slant * (y_range - mid_y)).astype(np.int32)
                
                # Check if the central segment is within image bounds
                central_x = x_indices[central_start:central_end]
                if np.any(central_x < 0) or np.any(central_x >= w):
                    continue
                
                # Extract the central slanted segment from the mask
                central_mask = grey_mask[y_range[central_start:central_end], central_x]
                
                if np.sum(central_mask) < central_height * 0.85:
                    continue
                
                # Color consistency check for the central segment
                segment_colors = img_lab[y_range[central_start:central_end], central_x]
                std_dev = np.std(segment_colors, axis=0)
                
                if np.all(std_dev < threshold):
                    avg_color = np.mean(segment_colors, axis=0)
                    candidates.append((float(x0), float(central_height), avg_color))

    if not candidates: return [], []
    
    # Cluster nearby candidate lines and pick the best one from each cluster
    candidates.sort(key=lambda x: x[0])
    final = []
    curr = [candidates[0]]
    for i in range(1, len(candidates)):
        if candidates[i][0] - curr[-1][0] < 5:
            curr.append(candidates[i])
        else:
            best = max(curr, key=lambda x: x[1])
            final.append((sum(c[0] for c in curr)/len(curr), best[1], best[2]))
            curr = [candidates[i]]
    best = max(curr, key=lambda x: x[1])
    final.append((sum(c[0] for c in curr)/len(curr), best[1], best[2]))
    return final, candidates

def detect_selected_cells(image_path, grid_boxes, debug_manager=None):
    """
    Checks each grid box to see if it contains a 'selected' badge (blue checkmark)
    or a 'loading' spinner.
    Returns (list of selected indices, list of loading indices).
    """
    img = cv2.imread(image_path)
    if img is None: return [], []
    sel, ld = [], []
    badge_rgb = (27, 115, 232) # Standard reCAPTCHA blue
    for i, box in enumerate(grid_boxes):
        cell = img[box[1]:box[3], box[0]:box[2]]
        if cell.size == 0: continue
        
        # Check top-left corner for the selection badge
        tl = cell[0:int(cell.shape[0]*0.4), 0:int(cell.shape[1]*0.4)]
        if tl.size > 0 and _has_badge(tl, badge_rgb): 
            sel.append(i+1)
            continue
            
        # Check center for loading spinner
        cntr = cell[int(cell.shape[0]*0.3):int(cell.shape[0]*0.7), int(cell.shape[1]*0.3):int(cell.shape[1]*0.7)]
        if cntr.size > 0 and _is_loading(cntr, badge_rgb): 
            ld.append(i+1)
    return sel, ld

def _has_badge(roi, rgb):
    """
    Uses color segmentation and shape analysis to detect the reCAPTCHA 
    selection badge (a blue circle/checkmark).
    """
    mask = _create_delta_e_mask(roi, rgb, 8.0)
    # Check if there's enough blue color
    if cv2.countNonZero(mask) <= roi.size * 0.003: return False
    
    # Shape analysis: look for a circular-ish contour
    contours, _ = cv2.findContours(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    area, perim = cv2.contourArea(hull), cv2.arcLength(hull, True)
    
    # Circularity check
    if perim == 0 or (4*np.pi*area/(perim*perim)) < 0.8: return False
    
    # Position check: badge should be in the top-left portion of its ROI
    M = cv2.moments(hull)
    if M["m00"] == 0 or (M["m10"]/M["m00"]) > roi.shape[1]*0.7 or (M["m01"]/M["m00"]) > roi.shape[0]*0.7: return False
    return True

def _is_loading(roi, rgb):
    """
    Detects if a cell is in a 'loading' state based on the amount 
    of blue color in the center.
    """
    mask = _create_delta_e_mask(roi, rgb, 12.0)
    # Loading state usually has a specific range of blue pixels
    return 0.05 < (cv2.countNonZero(mask) / (roi.shape[0]*roi.shape[1])) < 0.6 if roi.size > 0 else False

def _create_delta_e_mask(img, rgb, thr):
    """
    Creates a binary mask of pixels that are within a certain 
    perceptual distance (Delta E) from the target RGB color.
    """
    t_lab = cv2.cvtColor(np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0]
    diff = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) - t_lab.astype(np.float32)
    return (np.sqrt(np.sum(diff**2, axis=2)) <= thr).astype(np.uint8)*255

def get_numbered_grid_overlay(image_path, grid_boxes, output_path=None):
    """
    Generates a debug image with numbered boxes overlaid on the original image.
    Uses high-visibility red labels with white text in the top-right.
    """
    ov = [{"bbox": [b[0], b[1], b[2]-b[0], b[3]-b[1]], "number": i+1, "color": "#FF0000", "box_style": "solid"} for i, b in enumerate(grid_boxes)]
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf: output_path = tf.name
    add_overlays_to_image(image_path, ov, output_path=output_path, label_position="top-right")
    return output_path
