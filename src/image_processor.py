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
    def get_color_similarity_score(color1: Any, color2: Any, use_cie2000: bool = True, is_lab: bool = False) -> int:
        """
        Returns a similarity score from 1 to 100 (100 being identical).
        
        Uses CIELAB color space and Delta E formula. Optimized with math module.
        
        Args:
            color1: RGB or LAB tuple.
            color2: RGB or LAB tuple.
            use_cie2000: If True, use the complex CIEDE2000 formula (most accurate).
            is_lab: Set to True if colors are already in LAB space.
        """
        # Identity check
        if color1[0] == color2[0] and color1[1] == color2[1] and color1[2] == color2[2]:
            return 100

        if is_lab:
            l1, a1, b1 = color1
            l2, a2, b2 = color2
        else:
            # Convert RGB to LAB
            # OpenCV expects BGR and uint8 for cvtColor
            c1_bgr = np.array([[[color1[2], color1[1], color1[0]]]], dtype=np.uint8)
            c2_bgr = np.array([[[color2[2], color2[1], color2[0]]]], dtype=np.uint8)
            
            lab1 = cv2.cvtColor(c1_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
            lab2 = cv2.cvtColor(c2_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
            
            # Adjust LAB values to standard scales (L: 0-100, a,b: -128-127)
            l1, a1, b1 = lab1[0] * 100.0 / 255.0, lab1[1] - 128.0, lab1[2] - 128.0
            l2, a2, b2 = lab2[0] * 100.0 / 255.0, lab2[1] - 128.0, lab2[2] - 128.0
        
        if use_cie2000:
            delta_e = ImageProcessor.calculate_delta_e_cie2000((l1, a1, b1), (l2, a2, b2))
        else:
            # CIE76: Euclidean distance in LAB space (much faster)
            delta_e = math.sqrt((l1 - l2)**2 + (a1 - a2)**2 + (b1 - b2)**2)
            
        # Map Delta E to 1-100 score. 
        score = 100.0 - delta_e
        return int(max(1, min(100, round(score))))

    @staticmethod
    def calculate_delta_e_cie2000(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
        """
        Calculate CIEDE2000 Delta E.
        Highly optimized implementation using the math module for scalar performance.
        """
        L1, a1, b1 = lab1
        L2, a2, b2 = lab2

        # Step 1: Calculate C' and h'
        C1 = math.sqrt(a1**2 + b1**2)
        C2 = math.sqrt(a2**2 + b2**2)
        C_bar = (C1 + C2) / 2.0

        # Precompute C_bar^7
        C_bar7 = C_bar**7
        G = 0.5 * (1 - math.sqrt(C_bar7 / (C_bar7 + 6103515625))) # 25**7 = 6103515625

        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2

        C1_prime = math.sqrt(a1_prime**2 + b1**2)
        C2_prime = math.sqrt(a2_prime**2 + b2**2)

        def get_h_prime(b, a_p):
            if b == 0 and a_p == 0: return 0
            h = math.degrees(math.atan2(b, a_p))
            return h if h >= 0 else h + 360

        h1_prime = get_h_prime(b1, a1_prime)
        h2_prime = get_h_prime(b2, a2_prime)

        # Step 2: Calculate Delta L', Delta C', and Delta h'
        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime

        if C1_prime * C2_prime == 0:
            delta_h_prime = 0
        else:
            delta_h_prime = h2_prime - h1_prime
            if delta_h_prime > 180:
                delta_h_prime -= 360
            elif delta_h_prime < -180:
                delta_h_prime += 360

        delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2.0))

        # Step 3: Calculate S_L, S_C, S_H and T
        L_bar_prime = (L1 + L2) / 2.0
        C_bar_prime = (C1_prime + C2_prime) / 2.0

        if C1_prime * C2_prime == 0:
            h_bar_prime = h1_prime + h2_prime
        else:
            h_bar_prime = (h1_prime + h2_prime) / 2.0
            if abs(h1_prime - h2_prime) > 180:
                if h1_prime + h2_prime < 360:
                    h_bar_prime += 180
                else:
                    h_bar_prime -= 180

        h_bar_rad = math.radians(h_bar_prime)
        T = (1 - 0.17 * math.cos(h_bar_rad - math.radians(30))
               + 0.24 * math.cos(2 * h_bar_rad)
               + 0.32 * math.cos(3 * h_bar_rad + math.radians(6))
               - 0.20 * math.cos(4 * h_bar_rad - math.radians(63)))

        S_L = 1 + (0.015 * (L_bar_prime - 50)**2) / math.sqrt(20 + (L_bar_prime - 50)**2)
        S_C = 1 + 0.045 * C_bar_prime
        S_H = 1 + 0.015 * C_bar_prime * T

        # Step 4: Calculate R_T
        delta_theta = 30 * math.exp(-((h_bar_prime - 275) / 25)**2)
        C_bar_prime7 = C_bar_prime**7
        R_C = 2 * math.sqrt(C_bar_prime7 / (C_bar_prime7 + 6103515625))
        R_T = -math.sin(math.radians(2 * delta_theta)) * R_C

        # Step 5: Final calculation
        delta_E = math.sqrt(
            (delta_L_prime / S_L)**2 +
            (delta_C_prime / S_C)**2 +
            (delta_H_prime / S_H)**2 +
            R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
        )

        return delta_E

    # =========================================================================
    # Background Removal Logic (Stateful)
    # =========================================================================

    def remove_background(
        self, 
        image_path: str, 
        prompt: str,
        max_objects: int = 1,
    ) -> Optional[Image.Image]:
        """
        Remove background from an image by selecting the segment matching the prompt.
        Returns RGBA image with transparent background.
        
        Args:
            image_path: Path to the image
            prompt: Description of the object to keep (e.g., "a puzzle piece")
            max_objects: How many objects to consider (defaults to 1 for cleanest removal)
        """
        if not self.attention:
            raise RuntimeError("Background removal requires attention_extractor.")

        if self.debug:
            self.debug.log(f"Removing background for '{prompt}' using SAM 3...")
        
        # 1. Get mask with SAM 3
        try:
            masks = self.attention.get_mask(image_path, prompt, max_objects=max_objects)
        except Exception as e:
            if self.debug: self.debug.log(f"SAM 3 mask retrieval failed for background removal: {e}")
            return None

        if not masks:
            if self.debug: self.debug.log(f"Background removal: No masks found for prompt '{prompt}'.")
            return None

        # 2. Use the top match mask
        mask = masks[0] # (H, W) boolean numpy array
        
        if self.debug:
            self.debug.log(f"Selected mask for background removal with prompt '{prompt}'.")

        # 3. Create RGBA image
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGBA")
                width, height = img.size
                
                # Ensure mask matches image size
                if mask.shape != (height, width):
                    if self.debug: self.debug.log(f"Resizing mask from {mask.shape} to {(height, width)}")
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                    mask_img = mask_img.resize((width, height), resample=Image.NEAREST)
                    mask_uint8 = np.array(mask_img)
                else:
                    mask_uint8 = (mask * 255).astype(np.uint8)
                
                # Fill holes in the mask to preserve internal details
                # RETR_EXTERNAL retrieves only the extreme outer contours
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled_mask = np.zeros_like(mask_uint8)
                cv2.drawContours(filled_mask, contours, -1, 255, cv2.FILLED)
                
                alpha_img = Image.fromarray(filled_mask, mode='L')
                
                out_img = img.copy()
                out_img.putalpha(alpha_img)
                
                # Save for debug if manager is present
                if self.debug:
                    ext = os.path.splitext(image_path)[1] or ".png"
                    with tempfile.NamedTemporaryFile(suffix=f"_bg_result{ext}", delete=False) as tf:
                        debug_out = tf.name
                    self._temp_files.append(debug_out)
                    out_img.save(debug_out)
                    self.debug.save_image(debug_out, f"bg_removal_result_{os.path.basename(image_path)}")
                
                return out_img
        except Exception as e:
            if self.debug: self.debug.log(f"Background removal creation error: {e}")
                
        return None

