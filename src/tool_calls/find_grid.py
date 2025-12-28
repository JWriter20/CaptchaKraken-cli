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
                    score = center_offset * 1000
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
                            score = rel_diff * 100 + center_offset * 1000
                            candidates[3].append(([p1, p2, p3], score, (d1 + d2) / 2.0, min(l1, l2, l3), [c1, c2, c3]))

        # Sort candidates by score (lower is better)
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
                                s_diff = abs(hd - vd) / max(hd, vd)
                                score = hs_sc + vs_sc + s_diff * 1000 + abs(slant) * 500
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

    if slant_to_try is not None:
        best_grid, min_score = run_detection(slant_to_try, threshold=10.0, color_thr=12.0)
        return best_grid

    # Default logic: try no slant first, then others
    best_grid, min_score = run_detection(0.0, threshold=10.0, color_thr=12.0)
    
    # If no grid found, try various slant angles (typical for reCAPTCHA)
    if not best_grid:
        for slant in [0.015, -0.015]:
            grid, score = run_detection(slant, threshold=10.0, color_thr=12.0)
            if grid and score < min_score:
                min_score, best_grid = score, grid

    if not best_grid: return None
    
    # Final debug output: Save image with detected grid boxes
    if debug_manager and getattr(debug_manager, 'enabled', False):
        overlay = img.copy()
        for i, (x1, y1, x2, y2) in enumerate(best_grid):
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(overlay, str(i+1), (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        image_basename = os.path.basename(image_path)
        # Use slant in filename if provided or if we found one
        slant_val = slant_to_try if slant_to_try is not None else 0.0 # simplified
        slant_str = f"{slant_val:.3f}".replace('.', '_')
        cv2.imwrite(os.path.join(str(getattr(debug_manager, 'base_dir', ".")), f"grid_final_slant_{slant_str}_{image_basename}"), overlay)
        
    return best_grid

def _generate_slanted_grid(size, hs, vs, hd, vd, h, w, slant):
    """
    Calculates the bounding boxes for each cell in the grid, 
    accounting for the slant factor.
    """
    mid_x, mid_y = w / 2, h / 2
    # Extrapolate grid boundaries from detected internal lines
    if size == 4:
        y_bounds = [hs[1] - 2*hd, hs[1] - hd, hs[1], hs[1] + hd, hs[1] + 2*hd]
        x_bounds = [vs[1] - 2*vd, vs[1] - vd, vs[1], vs[1] + vd, vs[1] + 2*vd]
    else:
        my = (hs[0] + hs[1]) / 2
        mx = (vs[0] + vs[1]) / 2
        y_bounds = [my - 1.5*hd, my - 0.5*hd, my + 0.5*hd, my + 1.5*hd]
        x_bounds = [mx - 1.5*vd, mx - 0.5*vd, mx + 0.5*vd, mx + 1.5*vd]

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
    grey_mask = (img_lab[:, :, 0] > 50) & (chroma < 5.5)
    grey_mask_u8 = grey_mask.astype(np.uint8) * 255

    candidates = []
    
    if axis == 1: # Horizontal-ish
        # Intelligent selective check: check center for potential lines
        center_x = w // 2
        cx_start, cx_end = max(0, center_x - 1), min(w, center_x + 2)
        # Note: we use the grey_mask directly here, but we will scan along slanted paths
        possible_y0s = np.where(np.any(grey_mask[:, cx_start:cx_end] > 0, axis=1))[0]
        
        if len(possible_y0s) > 0:
            width_thr = w * 0.7
            x_range = np.arange(w)
            
            for y0 in possible_y0s:
                # Slanted scanning: calculate the Y coordinates along a slanted line starting at y0
                # y_src = y0 + slant * (x - mid_x)
                y_indices = np.round(y0 + slant * (x_range - mid_x)).astype(np.int32)
                
                # Filter out-of-bounds indices
                valid_mask = (y_indices >= 0) & (y_indices < h)
                if np.sum(valid_mask) < width_thr: continue
                
                # Extract slanted row segment from the mask
                slanted_row = np.zeros(w, dtype=bool)
                slanted_row[valid_mask] = grey_mask[y_indices[valid_mask], x_range[valid_mask]]
                
                if np.sum(slanted_row) < width_thr: continue
                
                # Find longest continuous run in the slanted row
                padded = np.zeros(w + 2, dtype=np.uint8)
                padded[1:-1] = slanted_row.astype(np.uint8)
                diff = np.diff(padded.astype(np.int16))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                if len(starts) == 0: continue
                lengths = ends - starts
                max_idx = np.argmax(lengths)
                max_len = lengths[max_idx]
                
                if max_len >= width_thr:
                    start, end = starts[max_idx], ends[max_idx]
                    # Extract the actual color pixels along the slanted path for consistency check
                    segment_y = y_indices[start:end]
                    segment_x = x_range[start:end]
                    segment_colors = img_lab[segment_y, segment_x]
                    
                    # Consistency check: Std dev of colors should be low
                    std_dev = np.std(segment_colors, axis=0)
                    if np.all(std_dev < threshold):
                        avg_color = np.mean(segment_colors, axis=0)
                        candidates.append((float(y0), float(max_len), avg_color))
    else: # Vertical-ish
        # Center check: use a 3-pixel window for robustness
        center_y = h // 2
        cy_start, cy_end = max(0, center_y - 1), min(h, center_y + 2)
        possible_x0s = np.where(np.any(grey_mask[cy_start:cy_end, :], axis=0))[0]
        
        if len(possible_x0s) > 0:
            height_thr = h * 0.4
            y_range = np.arange(h)
            
            for x0 in possible_x0s:
                # Slanted scanning for vertical lines
                # x_src = x0 + slant * (y - mid_y)
                x_indices = np.round(x0 + slant * (y_range - mid_y)).astype(np.int32)
                
                valid_mask = (x_indices >= 0) & (x_indices < w)
                if np.sum(valid_mask) < height_thr: continue
                
                slanted_col = np.zeros(h, dtype=bool)
                slanted_col[valid_mask] = grey_mask[y_range[valid_mask], x_indices[valid_mask]]
                
                if np.sum(slanted_col) < height_thr: continue
                
                padded = np.zeros(h + 2, dtype=np.uint8)
                padded[1:-1] = slanted_col.astype(np.uint8)
                diff = np.diff(padded.astype(np.int16))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                if len(starts) == 0: continue
                lengths = ends - starts
                max_idx = np.argmax(lengths)
                max_len = lengths[max_idx]
                
                if max_len >= height_thr:
                    start, end = starts[max_idx], ends[max_idx]
                    segment_x = x_indices[start:end]
                    segment_y = y_range[start:end]
                    segment_colors = img_lab[segment_y, segment_x]
                    
                    std_dev = np.std(segment_colors, axis=0)
                    if np.all(std_dev < threshold):
                        avg_color = np.mean(segment_colors, axis=0)
                        candidates.append((float(x0), float(max_len), avg_color))

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
    """
    ov = [{"bbox": [b[0], b[1], b[2]-b[0], b[3]-b[1]], "number": i+1, "color": "#00FF00", "box_style": "solid"} for i, b in enumerate(grid_boxes)]
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf: output_path = tf.name
    add_overlays_to_image(image_path, ov, output_path=output_path, label_position="top-right")
    return output_path
