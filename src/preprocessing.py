"""
Preprocessing utilities for improving image detection.
"""
import numpy as np
import cv2
from PIL import Image

def blend_colors(image_path: str, output_path: str, spatial_radius: int = 10, color_radius: int = 20) -> str:
    """
    Apply mean shift filtering to blend similar colors and flatten texture.
    This creates a "cartoon" effect that can help object detection.
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
        spatial_radius: Spatial window radius
        color_radius: Color window radius
        
    Returns:
        Path to processed image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    # Mean shift filtering
    filtered = cv2.pyrMeanShiftFiltering(img, sp=spatial_radius, sr=color_radius)
    
    cv2.imwrite(output_path, filtered)
    return output_path

def sharpen_edges(image_path: str, output_path: str) -> str:
    """
    Apply unsharp masking to sharpen edges.
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
        
    Returns:
        Path to processed image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
    unsharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0, img)
    
    cv2.imwrite(output_path, unsharp)
    return output_path

