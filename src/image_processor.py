import os
import tempfile
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any

class ImageProcessor:
    """
    Unified image processing class handling:
    - Basic image manipulation (sharpen, blur, contrast, etc.)
    - Grid detection
    - Background removal (using segmentation + LLM)
    """
    
    def __init__(self, attention_extractor=None, planner=None, debug_manager=None):
        """
        Args:
            attention_extractor: AttentionExtractor instance (required for background removal)
            planner: ActionPlanner instance (required for background removal)
            debug_manager: DebugManager instance (optional)
        """
        self.attention = attention_extractor
        self.planner = planner
        self.debug = debug_manager
        self._temp_files: List[str] = []

    def __del__(self):
        # Cleanup temp files
        for f in self._temp_files:
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception:
                    pass

    # =========================================================================
    # Static Utility Methods (Stateless)
    # =========================================================================

    @staticmethod
    def to_greyscale(image_path: str, output_path: str) -> None:
        """Convert an image to greyscale and save it."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_path, gray)

    @staticmethod
    def to_edge_outline(image_path: str, output_path: str, low_threshold: int = 15, high_threshold: int = 60) -> None:
        """Detect and isolate edges in an image using Canny edge detection."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        cv2.imwrite(output_path, edges)

    @staticmethod
    def sharpen_image(image_path: str, output_path: str) -> None:
        """Sharpen an image using a generic kernel."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(output_path, sharpened)

    @staticmethod
    def sharpen_edges(image_path: str, output_path: str) -> str:
        """Apply unsharp masking to sharpen edges (alternative to sharpen_image)."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
        unsharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0, img)
        
        cv2.imwrite(output_path, unsharp)
        return output_path

    @staticmethod
    def apply_clahe(image_path: str, output_path: str, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> None:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        cv2.imwrite(output_path, final)

    @staticmethod
    def apply_contrast_enhancement(image_path: str, output_path: str, alpha: float = 1.5, beta: int = 0) -> None:
        """Apply contrast enhancement using linear stretching."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        cv2.imwrite(output_path, enhanced)

    @staticmethod
    def apply_gaussian_blur(image_path: str, output_path: str, kernel_size: Tuple[int, int] = (5, 5), sigma_x: float = 0) -> None:
        """Apply Gaussian Blur to an image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        blurred = cv2.GaussianBlur(img, kernel_size, sigma_x)
        cv2.imwrite(output_path, blurred)

    @staticmethod
    def blend_colors(image_path: str, output_path: str, spatial_radius: int = 10, color_radius: int = 20) -> str:
        """Apply mean shift filtering to blend similar colors (cartoon effect)."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        filtered = cv2.pyrMeanShiftFiltering(img, sp=spatial_radius, sr=color_radius)
        cv2.imwrite(output_path, filtered)
        return output_path

    @staticmethod
    def merge_similar_colors(image_path: str, output_path: str, k: int = 8) -> None:
        """Quantize image colors to K clusters using K-Means."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized_img = quantized.reshape(img.shape)
        
        quantized_bgr = cv2.cvtColor(quantized_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, quantized_bgr)

    @staticmethod
    def get_grid_bounding_boxes(image_path: str) -> Optional[List[Tuple[int, int, int, int]]]:
        """Detects a grid of images separated by white spacing."""
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        h, w = binary.shape

        row_sums = np.sum(binary, axis=1) / 255
        row_ratio = row_sums / w
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
        if len(h_lines) < 3:
            return None

        possible_grids = []
        for i in range(len(h_lines)):
            for j in range(i + 2, len(h_lines) + 1):
                subset = h_lines[i:j]
                if len(subset) < 3:
                    continue
                spacings = [subset[k] - subset[k - 1] for k in range(1, len(subset))]
                avg_spacing = np.mean(spacings)
                std_spacing = np.std(spacings)
                if std_spacing < max(5, avg_spacing * 0.1):
                    n_rows = len(subset) - 1
                    if n_rows in [3, 4]:
                        possible_grids.append((subset, n_rows))

        valid_grids = []
        for lines, n_rows in possible_grids:
            y_min = lines[0]
            y_max = lines[-1]
            crop = binary[y_min:y_max, :]
            crop_h, crop_w = crop.shape
            col_sums = np.sum(crop, axis=0) / 255
            col_ratio = col_sums / crop_h
            v_indices = np.where(col_ratio > threshold_ratio)[0]
            v_lines_cand = group_indices(v_indices)
            v_lines_internal = [x for x in v_lines_cand if x > crop_w * 0.05 and x < crop_w * 0.95]
            n_cols = len(v_lines_internal) + 1
            
            is_valid = False
            if (n_rows, n_cols) in [(3, 3), (4, 4), (3, 4), (4, 3)]:
                is_valid = True

            if is_valid:
                valid_grids.append({
                    "h_lines": lines,
                    "v_lines_internal": v_lines_internal,
                    "rows": n_rows,
                    "cols": n_cols,
                    "area": (lines[-1] - lines[0]) * w,
                })

        if not valid_grids:
            return None

        best_grid = max(valid_grids, key=lambda x: (x["rows"] * x["cols"], x["area"]))
        h_lines_found = best_grid["h_lines"]
        v_lines_internal = best_grid["v_lines_internal"]
        rows = best_grid["rows"]
        cols = best_grid["cols"]

        if rows >= 3:
            internal_h_lines = h_lines_found[1:-1]
            if len(internal_h_lines) >= 2:
                spacings_h = [internal_h_lines[i] - internal_h_lines[i - 1] for i in range(1, len(internal_h_lines))]
                avg_h_spacing = np.mean(spacings_h)
            else:
                avg_h_spacing = internal_h_lines[-1] - internal_h_lines[0]
            new_top = int(internal_h_lines[0] - avg_h_spacing)
            new_bottom = int(internal_h_lines[-1] + avg_h_spacing)
            h_lines_found = [new_top] + list(internal_h_lines) + [new_bottom]

        if len(v_lines_internal) > 1:
            avg_v_spacing = np.mean([v_lines_internal[k] - v_lines_internal[k - 1] for k in range(1, len(v_lines_internal))])
        else:
            avg_v_spacing = w / cols

        left_border = max(0, int(v_lines_internal[0] - avg_v_spacing))
        right_border = min(w, int(v_lines_internal[-1] + avg_v_spacing))
        x_coords = [left_border] + v_lines_internal + [right_border]
        
        if len(x_coords) != cols + 1:
            x_step = (right_border - left_border) / cols
            x_coords = [int(left_border + k * x_step) for k in range(cols + 1)]

        boxes = []
        for r in range(rows):
            for c in range(cols):
                y1 = int(h_lines_found[r])
                y2 = int(h_lines_found[r + 1])
                x1 = int(x_coords[c])
                x2 = int(x_coords[c + 1])
                boxes.append((x1, y1, x2, y2))

        return boxes

    def detect_selected_cells(self, image_path: str, grid_boxes: List[Tuple[int, int, int, int]]) -> Tuple[List[int], List[int]]:
        """
        Detects grid cells status using Computer Vision.
        Returns a tuple of two lists:
        1. selected_indices: Cells with a checkmark badge (Completed selection).
        2. loading_indices: Cells with a spinner or center checkmark (Loading/Fading).

        Differentiation:
        - Selected: Small circular badge with checkmark in Top-Left.
        - Loading: Spinner or Large Badge with checkmark in the Center.
        """
        img = cv2.imread(image_path)
        if img is None:
            return [], []

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        selected_indices = []
        loading_indices = []

        # Color ranges
        # ReCaptcha Blue - Tuned tightly for rgb(27, 115, 232) -> hsv(107, 225, 232)
        # Standard Blue is ~108. User requested tight margin.
        # Widen slightly to cover 108 +/- 4 roughly, and lower saturation support.
        lower_blue = np.array([102, 130, 130])
        upper_blue = np.array([114, 255, 255])
        # hCaptcha Green
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255]) 

        for i, box in enumerate(grid_boxes):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            # --- 1. Check Top-Left for Selection Badge ---
            # ROI: Top-Left 25% (Split into 4, check top-left)
            # Stricter ROI to avoid catching content in the cell
            tl_w = int(w * 0.25)
            tl_h = int(h * 0.25)
            tl_w = max(1, min(tl_w, w))
            tl_h = max(1, min(tl_h, h))
            
            tl_roi_hsv = hsv[y1:y1+tl_h, x1:x1+tl_w]
            tl_bgr = img[y1:y1+tl_h, x1:x1+tl_w]
            
            is_selected = False
            
            if tl_roi_hsv.size > 0:
                # Blue Badge
                mask_blue = cv2.inRange(tl_roi_hsv, lower_blue, upper_blue)
                blue_count = cv2.countNonZero(mask_blue)
                threshold = (tl_w * tl_h) * 0.03
                
                if blue_count > threshold:
                    if self._check_contours_for_badge(mask_blue, tl_bgr):
                        is_selected = True
                        if self.debug: self.debug.log(f"Cell {i+1}: Detected Blue Selection Badge")
                    elif self.debug:
                        self.debug.log(f"Cell {i+1}: Blue pixels found ({blue_count} > {threshold:.1f}) but badge check failed.")

                # Green Badge
                if not is_selected:
                    mask_green = cv2.inRange(tl_roi_hsv, lower_green, upper_green)
                    green_count = cv2.countNonZero(mask_green)
                    if green_count > threshold:
                        if self._check_contours_for_badge(mask_green, tl_bgr):
                            is_selected = True
                            if self.debug: self.debug.log(f"Cell {i+1}: Detected Green Selection Badge")
                        elif self.debug:
                            self.debug.log(f"Cell {i+1}: Green pixels found ({green_count} > {threshold:.1f}) but badge check failed.")
            
            if is_selected:
                selected_indices.append(i + 1)
                # If selected, we assume it's not "loading" in a way that blocks progress
                # (Static selection)
                continue 

            # --- 2. Check Center for Loading Spinner / Badge ---
            # ROI: Center 40%
            cx = int(w * 0.3)
            cy = int(h * 0.3)
            cw = int(w * 0.4)
            ch = int(h * 0.4)
            
            center_roi_hsv = hsv[y1+cy:y1+cy+ch, x1+cx:x1+cx+cw]
            center_roi_bgr = img[y1+cy:y1+cy+ch, x1+cx:x1+cx+cw]
            
            if center_roi_hsv.size > 0:
                # Loading indicators are typically Blue
                mask_blue_center = cv2.inRange(center_roi_hsv, lower_blue, upper_blue)
                blue_density = cv2.countNonZero(mask_blue_center) / (cw * ch)
                
                # Check for "Large blue circle with checkmark" (Badge in center)
                if blue_density > 0.05:
                     # Check if it's a badge (Checkmark)
                     if self._check_contours_for_badge(mask_blue_center, center_roi_bgr):
                         loading_indices.append(i + 1)
                         if self.debug: self.debug.log(f"Cell {i+1}: Detected Center Loading Badge (Checkmark)")
                     
                     # Or if it's just a spinner (significant blue mass in center)
                     else:
                         # Refined Spinner Detection:
                         # Spinners are ring-like (low density in bbox) and roughly square aspect ratio.
                         # Blue signs are solid (high density) and/or rectangular.
                        contours, _ = cv2.findContours(mask_blue_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            max_cnt = max(contours, key=cv2.contourArea)
                            x, y, w_cnt, h_cnt = cv2.boundingRect(max_cnt)
                            
                            if w_cnt > 0 and h_cnt > 0:
                                aspect_ratio = float(w_cnt) / h_cnt
                                shape_area = w_cnt * h_cnt
                                # Count pixels within the bounding box (approximation using total blue pixels)
                                blue_pixels = cv2.countNonZero(mask_blue_center)
                                shape_density = blue_pixels / shape_area
                                
                                # Check size relative to ROI (Spinner should cover a good portion of the center)
                                rel_width = w_cnt / cw
                                rel_height = h_cnt / ch

                                # Criteria for Spinner:
                                # 1. Aspect Ratio ~ 1 (Circle) - Stricter range
                                # 2. Shape Density < 0.6 (Ring structure) but > 0.25 (Not too sparse)
                                # 3. Size relative to ROI > 0.35 (Must be substantial)
                                if 0.8 < aspect_ratio < 1.25 and 0.25 < shape_density < 0.6 and rel_width > 0.35 and rel_height > 0.35:
                                    loading_indices.append(i + 1)
                                    if self.debug: self.debug.log(f"Cell {i+1}: Detected Center Loading Spinner (Density: {shape_density:.2f}, AR: {aspect_ratio:.2f}, RelSize: {rel_width:.2f}x{rel_height:.2f})")
                                elif self.debug:
                                    self.debug.log(f"Cell {i+1}: Ignored blue object (Density: {shape_density:.2f}, AR: {aspect_ratio:.2f}, RelSize: {rel_width:.2f}x{rel_height:.2f}) - likely a sign/static object")

        return selected_indices, loading_indices

    def _check_contours_for_badge(self, mask, roi_bgr) -> bool:
        """Helper to validate if a mask looks like a checkmark badge."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
            
        max_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_cnt)
        perimeter = cv2.arcLength(max_cnt, True)
        
        if perimeter == 0:
            return False
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Check for white checkmark INSIDE the badge bounding box
        # The mask only captures the blue/green part, excluding the white checkmark (saturation=0)
        x, y, w, h = cv2.boundingRect(max_cnt)
        
        # Extract the region corresponding to the badge from the ROI
        badge_roi = roi_bgr[y:y+h, x:x+w]
        
        if badge_roi.size == 0:
            return False

        badge_gray = cv2.cvtColor(badge_roi, cv2.COLOR_BGR2GRAY)
        # Check for bright pixels (white checkmark)
        _, white_mask = cv2.threshold(badge_gray, 150, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(white_mask)
        
        # Criteria: High circularity (badge shape) AND contains white pixels (checkmark)
        # User requirement: Circularity > 0.85
        # User requirement: Only white and blue pixels (handled by contour/mask check mostly, but can be stricter)
        
        passed = circularity > 0.85 and white_pixels > 2
        
        # Additional Check: Pixel purity
        # Ensure the badge area is mostly blue/white.
        # We can check the mask coverage vs bounding rect.
        # A circle fills pi/4 (approx 0.785) of a square.
        # If it's a solid circle, extent should be high.
        
        if passed:
             # Check if the content inside the contour is mostly the badge color + white checkmark
             # Extract ROI of the contour
             mask_roi = mask[y:y+h, x:x+w]
             # Count non-zero in mask (blue pixels)
             blue_px = cv2.countNonZero(mask_roi)
             # Total pixels in rect
             total_px = w * h
             # Verify it's not just a thin ring or noise
             if blue_px / total_px < 0.5: # Badge should be solid
                  passed = False
                  if hasattr(self, 'debug') and self.debug:
                       self.debug.log(f"Badge check failed: Low density ({blue_px/total_px:.2f})")

        if not passed and hasattr(self, 'debug') and self.debug:
             self.debug.log(f"Badge check failed: Circ={circularity:.2f}, WhitePx={white_pixels}")
        return passed

    @staticmethod
    def detect_checkbox(image_path: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects a checkbox in the image using lightweight computer vision techniques.
        Returns the (x, y, w, h) of the best candidate or None.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple thresholding - efficient and works well for high contrast line art like checkboxes
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = img.shape[:2]
        image_area = width * height
        
        candidates = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = float(w) / h
            
            # Filter based on properties
            
            # 1. Aspect ratio should be close to 1 (square)
            if not (0.8 < aspect_ratio < 1.2):
                continue
                
            # 2. Minimum absolute size (to avoid noise/text periods)
            if w < 20 or h < 20:
                continue

            # 3. Area relative to image size
            if not (0.001 * image_area < area < 0.05 * image_area):
                continue
                
            # 4. Solidity/Extent
            contour_area = cv2.contourArea(cnt)
            extent = contour_area / area
            if extent < 0.8: # Stricter extent to avoid complex shapes
                continue

            # 5. Content check - Checkboxes are usually empty (white/solid)
            # Crop margin to avoid border
            roi = gray[y+5:y+h-5, x+5:x+w-5] 
            if roi.size > 0:
                std_dev = np.std(roi)
                if std_dev > 35.0: # High variance means complex content (image)
                    continue

            candidates.append((x, y, w, h))
            
        # Heuristic: Checkbox captchas typically have 1 single checkbox.
        # If we find many candidates, it's likely a grid or something else.
        if len(candidates) > 2:
            return None
            
        if not candidates:
            return None
            
        # Return the largest candidate (most likely the main checkbox)
        best_candidate = max(candidates, key=lambda c: c[2] * c[3])
        return best_candidate

    # =========================================================================
    # Background Removal Logic (Stateful)
    # =========================================================================

    def remove_background(
        self, 
        image_path: str, 
        prompt: str,
        k: int = 3,
        merge_components: bool = True,
        min_area_ratio: float = 0.01,
        pre_merge_colors: bool = False
    ) -> Optional[Image.Image]:
        """
        Remove background from an image by selecting the segment matching the prompt.
        Returns RGBA image with transparent background.
        """
        if not self.attention:
            raise RuntimeError("Background removal requires attention_extractor.")

        if self.debug:
            self.debug.log(f"Removing background for '{prompt}' using color segmentation...")
        
        masks = []
        process_path = image_path

        # 1. Optional Preprocessing
        if pre_merge_colors:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                quant_path = tf.name
            self._temp_files.append(quant_path)
            try:
                self.merge_similar_colors(image_path, quant_path, k=5)
                process_path = quant_path
            except Exception as e:
                if self.debug: self.debug.log(f"Pre-merge colors failed: {e}")

        # 2. Segment
        try:
            res = self.attention.segment_by_color(
                process_path, 
                k=k, 
                merge_components=merge_components, 
                min_area_ratio=min_area_ratio
            )
            masks = res.get("masks", [])
        except Exception as e:
            if self.debug: self.debug.log(f"Color segmentation failed: {e}")
            return None

        if not masks:
            if self.debug: self.debug.log("Background removal: No masks found.")
            return None

        # 3. Visualize masks (Debug only)
        if self.debug:
            ext = os.path.splitext(image_path)[1] or ".png"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                candidates_path = tf.name
            self._temp_files.append(candidates_path)
            
            self.attention.visualize_masks(process_path, masks, output_path=candidates_path, draw_ids=True)
            self.debug.save_image(candidates_path, f"bg_removal_candidates_{os.path.basename(image_path)}")
            self.debug.log(f"Evaluating {len(masks)} masks with deterministic criteria for prompt: '{prompt}'")
            
        # 4. Deterministic Selection
        best_idx = -1
        max_score = -float('inf')
        
        with Image.open(process_path) as img:
            W, H = img.size
            
        center_x, center_y = W / 2.0, H / 2.0
        max_dist = np.sqrt(center_x**2 + center_y**2) or 1.0
        total_pixels = W * H
        
        # 10% margin for edge penalty
        margin_x = int(W * 0.1)
        margin_y = int(H * 0.1)
        
        for idx, m in enumerate(masks):
            if len(m.shape) == 3: 
                m = m[0]
            
            # Ensure boolean
            mask_bool = m > 0
            area = np.sum(mask_bool)
            
            if area == 0:
                continue
            
            # 1. Area Score (Larger is better)
            norm_area = area / total_pixels
            
            # 2. Centrality Score (Closer to center is better)
            y_indices, x_indices = np.where(mask_bool)
            if len(x_indices) == 0:
                continue
                
            centroid_x = np.mean(x_indices)
            centroid_y = np.mean(y_indices)
            
            dist = np.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2)
            norm_dist = dist / max_dist
            centrality_score = 1.0 - norm_dist
            
            # 3. Edge Penalty (More content near edge is bad)
            in_margin = (
                (x_indices < margin_x) | 
                (x_indices >= W - margin_x) | 
                (y_indices < margin_y) | 
                (y_indices >= H - margin_y)
            )
            edge_pixel_count = np.sum(in_margin)
            edge_ratio = edge_pixel_count / area
            
            # Weights
            w_area = 1.0
            w_center = 0.5
            w_edge = 5.0
            
            score = (w_area * norm_area) + (w_center * centrality_score) - (w_edge * edge_ratio)
            
            if self.debug:
                self.debug.log(f"Mask {idx}: Area={norm_area:.2f}, Cent={centrality_score:.2f}, Edge={edge_ratio:.2f} -> Score={score:.2f}")
            
            if score > max_score:
                max_score = score
                best_idx = idx
                
        selected_ids = [best_idx] if best_idx != -1 else []
        if self.debug and selected_ids:
            self.debug.log(f"Selected mask ID: {best_idx} (Score: {max_score:.2f})")

        if selected_ids:
            # 5. Create RGBA image with combined masks
            if len(masks) > 0:
                base_shape = masks[0].shape
                if len(base_shape) == 3: base_shape = base_shape[:2]
                combined_mask = np.zeros(base_shape, dtype=bool)
                
                for idx in selected_ids:
                    if 0 <= idx < len(masks):
                        m = masks[idx]
                        if len(m.shape) == 3: m = m[0]
                        combined_mask = np.logical_or(combined_mask, m)
                
                try:
                    with Image.open(image_path) as img:
                        img = img.convert("RGBA")
                        
                        # Fill holes in the mask to preserve internal details
                        mask_uint8 = (combined_mask * 255).astype(np.uint8)
                        # RETR_EXTERNAL retrieves only the extreme outer contours
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        filled_mask = np.zeros_like(mask_uint8)
                        cv2.drawContours(filled_mask, contours, -1, 255, cv2.FILLED)
                        
                        alpha_img = Image.fromarray(filled_mask, mode='L')
                        
                        if alpha_img.size != img.size:
                            alpha_img = alpha_img.resize(img.size, resample=Image.NEAREST)
                        
                        out_img = img.copy()
                        out_img.putalpha(alpha_img)
                        
                        # Save for debug
                        debug_out = candidates_path.replace(ext, "_result.png")
                        out_img.save(debug_out)
                        if self.debug:
                            self.debug.save_image(debug_out, f"bg_removal_result_{os.path.basename(image_path)}")
                        
                        return out_img
                except Exception as e:
                    if self.debug: self.debug.log(f"Background removal creation error: {e}")
                
        return None

