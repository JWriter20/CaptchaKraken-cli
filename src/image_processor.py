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
    def detect_movement(image1_path: str, image2_path: str, threshold: float = 0.005) -> bool:
        """
        Compare two images and return True if they are significantly different.
        
        Args:
            image1_path: Path to the first image.
            image2_path: Path to the second image.
            threshold: Percentage of pixels that must change to be considered movement.
        """
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        if img1 is None or img2 is None:
            return False
            
        if img1.shape != img2.shape:
            # If resolution changed, something definitely changed
            return True
            
        # Compute absolute difference
        diff = cv2.absdiff(img1, img2)
        # Convert to grayscale to simplify
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold the difference to ignore minor noise (compression artifacts, etc.)
        # 30 is a reasonable threshold for significant pixel change
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of changed pixels
        changed_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        change_ratio = changed_pixels / total_pixels
        
        return change_ratio > threshold

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

    # --- Configuration ---
    # Color detection using CIELAB color space with Delta E (perceptually uniform)
    # Uses CIE1976 Delta E (Euclidean distance in Lab space) for perceptually uniform color matching
    # Target blue: rgb(27, 115, 232) - the exact blue badge color
    # Delta E thresholds (CIE1976):
    #   < 1: Imperceptible difference
    #   1-2: Barely noticeable
    #   2-10: Noticeable but "close" for matching
    # Note: CIE1976 is simpler than CIEDE2000 but still provides good perceptual uniformity
    COLORS = {
        # Target blue badge color in RGB (will be converted to Lab)
        # Tight "badge blue" core - only very perceptually similar blues
        'BADGE_BLUE': {'target_rgb': (27, 115, 232), 'max_delta_e': 8.0},
        # Broader blue for spinners/loading indicators
        'BLUE':       {'target_rgb': (27, 115, 232), 'max_delta_e': 12.0},
    }

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

        selected_indices = []
        loading_indices = []
        
        # Extract base filename for debug image naming
        image_basename = os.path.splitext(os.path.basename(image_path))[0] if image_path else "unknown"

        if self.debug:
            self.debug.log(f"Starting CV detection for {len(grid_boxes)} cells")

        for i, (x1, y1, x2, y2) in enumerate(grid_boxes):
            cell_bgr = img[y1:y2, x1:x2]
            cell_id = i + 1

            # Save full cell image with ROI overlays for debugging
            if self.debug:
                # Create overlay showing ROI regions
                overlay = cell_bgr.copy()
                h, w = cell_bgr.shape[:2]
                
                # Draw top-left ROI rectangle (40% x 40%)
                tl_w = int(w * 0.4)
                tl_h = int(h * 0.4)
                cv2.rectangle(overlay, (0, 0), (tl_w, tl_h), (0, 255, 0), 2)  # Green for badge detection
                cv2.putText(overlay, "TL Badge", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw center ROI rectangle (40% x 40%, starting at 30%)
                cx = int(w * 0.3)
                cy = int(h * 0.3)
                cw = int(w * 0.4)
                ch = int(h * 0.4)
                cv2.rectangle(overlay, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 2)  # Blue for loading detection
                cv2.putText(overlay, "Center Loading", (cx + 5, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
            # 1. Check Top-Left (40% area) for a Selection Badge
            tl_roi_bgr = self._get_roi(cell_bgr, 0, 0, 0.4, 0.4)
            if tl_roi_bgr.size > 0:
                if self._has_badge(tl_roi_bgr, cell_id, tl_roi_bgr, image_basename):
                    selected_indices.append(cell_id)
                    if self.debug:
                        self.debug.log(f"Cell {cell_id}: Detected Selection Badge")
                continue

            # 2. Check Center (40% area) for Spinner/Loading Badge
            center_roi_bgr = self._get_roi(cell_bgr, 0.3, 0.3, 0.4, 0.4)
            if center_roi_bgr.size > 0:
                if self._is_loading(center_roi_bgr, cell_id, center_roi_bgr):
                    loading_indices.append(cell_id)
                    if self.debug:
                        self.debug.log(f"Cell {cell_id}: Detected Loading Indicator")

        if self.debug:
            self.debug.log(f"CV Detection complete: {len(selected_indices)} selected, {len(loading_indices)} loading")
            if selected_indices:
                self.debug.log(f"Selected cell IDs: {selected_indices}")
            if loading_indices:
                self.debug.log(f"Loading cell IDs: {loading_indices}")

        return selected_indices, loading_indices

    # --- Helper Logic ---

    def _take_debug_image(self, img, x_pct, y_pct, w_pct, h_pct, output_path: str):
        """Takes a debug image of a region of the image."""
        roi = self._get_roi(img, x_pct, y_pct, w_pct, h_pct)
        cv2.imwrite(output_path, roi)
        return output_path

    def _get_roi(self, img, x_pct, y_pct, w_pct, h_pct):
        """Extracts a sub-region of an image based on percentages."""
        h, w = img.shape[:2]
        y_start = int(h * y_pct)
        y_end = int(h * (y_pct + h_pct))
        x_start = int(w * x_pct)
        x_end = int(w * (x_pct + w_pct))
        return img[y_start:y_end, x_start:x_end]

    def _create_delta_e_mask(self, bgr_image, target_rgb: Tuple[int, int, int], max_delta_e: float) -> np.ndarray:
        """
        Creates a mask of pixels perceptually similar to target color using Delta E in CIELAB space.
        Uses CIE1976 Delta E (Euclidean distance in Lab space) for perceptually uniform color matching.
        This is faster than CIEDE2000 and still provides good perceptual uniformity.
        
        Args:
            bgr_image: BGR image (OpenCV format)
            target_rgb: Target color in RGB (0-255 range)
            max_delta_e: Maximum Delta E threshold (CIE1976)
            
        Returns:
            Binary mask (uint8, 0 or 255) where True indicates perceptually similar colors
        """
        # Convert target RGB to BGR for OpenCV
        target_bgr = (target_rgb[2], target_rgb[1], target_rgb[0])
        
        # Create a 1x1 image with the target color and convert to Lab
        target_bgr_array = np.array([[target_bgr]], dtype=np.uint8)
        target_lab = cv2.cvtColor(target_bgr_array, cv2.COLOR_BGR2LAB)[0, 0]
        
        # Convert the input image to Lab color space (vectorized, fast)
        image_lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        
        # Compute CIE1976 Delta E (Euclidean distance in Lab space)
        # Delta E = sqrt((L1-L2)^2 + (a1-a2)^2 + (b1-b2)^2)
        delta_e = np.sqrt(
            np.sum((image_lab.astype(np.float32) - target_lab.astype(np.float32)) ** 2, axis=2)
        )
        
        # Create mask where Delta E is within threshold
        mask = (delta_e <= max_delta_e).astype(np.uint8) * 255
        
        return mask

    def _has_badge(self, bgr_roi, cell_id: int = None, bgr_roi_debug = None, image_basename: str = "unknown") -> bool:
        """
        Checks if a region contains a Blue checkmark badge with circularity check.
        Uses CIELAB color space with Delta E (CIEDE2000) for perceptually uniform color matching.
        Requires: pixel count > threshold AND circularity > 0.85
        
        Args:
            bgr_roi: BGR region of interest
            cell_id: Cell identifier for debugging
            bgr_roi_debug: BGR region of interest for debug visualization (same as bgr_roi, kept for compatibility)
            image_basename: Base filename for debug image naming
        """
        color_name = 'BADGE_BLUE'
        color = self.COLORS[color_name]
        mask = self._create_delta_e_mask(bgr_roi, color['target_rgb'], color['max_delta_e'])
        pixel_count = cv2.countNonZero(mask)
        threshold = bgr_roi.size * 0.003
        
        # Save debug images if enabled and we have a cell_id
        debug_images = []
        if self.debug and cell_id is not None and bgr_roi_debug is not None:
            # 1. Save original ROI
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                roi_path = tf.name
            self._temp_files.append(roi_path)
            cv2.imwrite(roi_path, bgr_roi_debug)
            debug_images.append(("original_roi", roi_path))
            
            # 2. Save blue mask (color filter result)
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                mask_path = tf.name
            self._temp_files.append(mask_path)
            cv2.imwrite(mask_path, mask_colored)
            debug_images.append(("blue_mask", mask_path))
        
        # Check pixel count threshold first
        if pixel_count <= threshold:
            if self.debug and cell_id is not None:
                self.debug.log(f"Cell {cell_id}: Pixel count below threshold, {pixel_count} <= {threshold}")
                # Save debug images even on failure
                self._save_badge_debug_images(debug_images, image_basename, cell_id, "pixel_count_fail")
            return False
        
        # Morphology: close small gaps in the blue circle caused by anti-aliasing / compression.
        # IMPORTANT: the white checkmark should *not* be included in the mask (white has different BGR values), so
        # we compute circularity on the convex hull of the blue pixels to ignore the checkmark cut-out.
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        if self.debug and cell_id is not None and bgr_roi_debug is not None:
            # 3. Save cleaned mask (after morphology)
            cleaned_colored = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                cleaned_path = tf.name
            self._temp_files.append(cleaned_path)
            cv2.imwrite(cleaned_path, cleaned_colored)
            debug_images.append(("cleaned_mask", cleaned_path))

        # Check circularity - badge should be circular (on convex hull)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if self.debug and cell_id is not None:
                self.debug.log(f"Cell {cell_id}: No blue contours found")
                self._save_badge_debug_images(debug_images, image_basename, cell_id, "no_contours")
            return False
        
        # Find the largest contour (should be the badge)
        max_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_cnt)
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        
        # Compute centroid early for visualization
        h, w = bgr_roi.shape[:2]
        M = cv2.moments(hull)
        centroid_x = M["m10"] / M["m00"] if M["m00"] != 0 else 0
        centroid_y = M["m01"] / M["m00"] if M["m00"] != 0 else 0
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Create visualization with hull and centroid overlay early (for all subsequent checks)
        if self.debug and cell_id is not None and bgr_roi_debug is not None:
            # 4. Create visualization with convex hull and centroid
            viz = bgr_roi_debug.copy()
            # Draw convex hull
            cv2.drawContours(viz, [hull], -1, (0, 255, 0), 2)
            # Draw centroid (if valid)
            if M["m00"] != 0:
                cv2.circle(viz, (int(centroid_x), int(centroid_y)), 5, (0, 0, 255), -1)
            # Draw top-left region boundary (60% of width and height)
            cv2.rectangle(viz, (0, 0), (int(w * 0.6), int(h * 0.6)), (255, 255, 0), 1)
            # Add text annotations
            cv2.putText(viz, f"Circ: {circularity:.2f}", (5, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if M["m00"] != 0:
                cv2.putText(viz, f"Cent: ({int(centroid_x)}, {int(centroid_y)})", (5, h - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                viz_path = tf.name
            self._temp_files.append(viz_path)
            cv2.imwrite(viz_path, viz)
            debug_images.append(("analysis_viz", viz_path))

        # Area sanity check: badge should be a reasonably-sized blob within the ROI
        roi_area = bgr_roi.shape[0] * bgr_roi.shape[1]
        if roi_area > 0:
            area_ratio = area / roi_area
            if area_ratio < 0.05:
                if self.debug and cell_id is not None:
                    self.debug.log(f"Cell {cell_id}: Blue area too small ({area_ratio:.3f} of ROI)")
                    self._save_badge_debug_images(debug_images, image_basename, cell_id, "area_too_small")
                return False
        
        if perimeter == 0:
            if self.debug and cell_id is not None:
                self.debug.log(f"Cell {cell_id}: Zero perimeter")
                self._save_badge_debug_images(debug_images, image_basename, cell_id, "zero_perimeter")
            return False
        
        if self.debug and cell_id is not None:
            self.debug.log(f"Cell {cell_id}: Circularity = {circularity:.3f} (required: >0.85)")
        
        # Require circularity > 0.85 for badge detection
        if circularity < 0.85:
            if self.debug and cell_id is not None:
                self.debug.log(f"Cell {cell_id}: Circularity too low")
                self._save_badge_debug_images(debug_images, image_basename, cell_id, "circularity_fail")
            return False
        
        # Check that the badge is positioned in the top-left corner
        # Badges should be near the corner, not in the center or elsewhere
        if M["m00"] == 0:
            if self.debug and cell_id is not None:
                self.debug.log(f"Cell {cell_id}: Zero moment")
                self._save_badge_debug_images(debug_images, image_basename, cell_id, "zero_moment")
            return False
        
        # Badge should be in top-left region (first 60% of width and height)
        # Using 60% instead of 50% to account for slight positioning variations
        if centroid_x > w * 0.6 or centroid_y > h * 0.6:
            if self.debug and cell_id is not None:
                self.debug.log(f"Cell {cell_id}: Badge centroid ({centroid_x:.1f}, {centroid_y:.1f}) not in top-left region")
                self._save_badge_debug_images(debug_images, image_basename, cell_id, "centroid_fail")
            return False
        
        if self.debug and cell_id is not None:
            self.debug.log(f"Cell {cell_id}: Badge centroid ({centroid_x:.1f}, {centroid_y:.1f}) in top-left region")
            # Save debug images for successful detection
            self._save_badge_debug_images(debug_images, image_basename, cell_id, "success")
        
        return True  # Found a circular badge in the top-left corner
    
    def _save_badge_debug_images(self, debug_images: List[Tuple[str, str]], image_basename: str, cell_id: int, status: str):
        """
        Save debug images for badge detection analysis.
        
        Args:
            debug_images: List of (name, path) tuples for debug images
            image_basename: Base filename for naming
            cell_id: Cell identifier
            status: Status string (success, pixel_count_fail, etc.)
        """
        if not self.debug or not debug_images:
            return
        
        for name, path in debug_images:
            if os.path.exists(path):
                debug_name = f"badge_analysis_{image_basename}_cell{cell_id}_{name}_{status}.png"
                self.debug.save_image(path, debug_name)

    def _is_loading(self, bgr_roi, cell_id: int = None, bgr_roi_debug = None) -> bool:
        """
        Detects blue movement/spinners in the center of the cell.
        Uses CIELAB color space with Delta E (CIEDE2000) for perceptually uniform color matching.
        """
        color = self.COLORS['BLUE']
        blue_mask = self._create_delta_e_mask(bgr_roi, color['target_rgb'], color['max_delta_e'])
        pixel_count = cv2.countNonZero(blue_mask)
        total_pixels = bgr_roi.shape[0] * bgr_roi.shape[1]
        density = pixel_count / total_pixels if total_pixels > 0 else 0
            
        if self.debug and cell_id is not None:
            self.debug.log(f"Cell {cell_id}: Center blue density = {density:.3f} ({pixel_count}/{total_pixels} pixels)")
        
        return 0.05 < density < 0.60  # Spinners are rings, so density is moderate

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

