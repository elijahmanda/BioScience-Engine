"""
Utility functions for image processing and analysis
"""

import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Optional
from pathlib import Path


def load_image(filepath: str) -> np.ndarray:
    """
    Load a single microscopy image
    
    Args:
        filepath: Path to image file
        
    Returns:
        Image as numpy array (grayscale, float32, normalized to 0-1)
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")
    
    # Normalize to 0-1 range
    return img.astype(np.float32) / 255.0


def load_tiff_stack(filepath: str) -> List[np.ndarray]:
    """
    Load multi-page TIFF file
    
    Args:
        filepath: Path to TIFF file
        
    Returns:
        List of images
    """
    import tifffile
    
    img_stack = tifffile.imread(filepath)
    
    if img_stack.ndim == 2:
        return [img_stack.astype(np.float32) / 255.0]
    else:
        return [img.astype(np.float32) / 255.0 for img in img_stack]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-1 range
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    img_min = image.min()
    img_max = image.max()
    
    if img_max - img_min < 1e-10:
        return np.zeros_like(image)
    
    return (image - img_min) / (img_max - img_min)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, 
                tile_size: int = 8) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization
    
    Args:
        image: Input image (0-1 range)
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image
    """
    # Convert to uint8 for CLAHE
    img_uint8 = (image * 255).astype(np.uint8)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                            tileGridSize=(tile_size, tile_size))
    
    # Apply CLAHE
    enhanced = clahe.apply(img_uint8)
    
    return enhanced.astype(np.float32) / 255.0


def distance_between_points(p1: Tuple[float, float], 
                           p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        Distance
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.sqrt(dx * dx + dy * dy)


def compute_intensity_stats(image: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """
    Compute intensity statistics for an image
    
    Args:
        image: Input image
        mask: Optional mask (only compute stats in masked region)
        
    Returns:
        Dictionary of statistics
    """
    if mask is not None:
        pixels = image[mask > 0]
    else:
        pixels = image.flatten()
    
    return {
        'mean': np.mean(pixels),
        'median': np.median(pixels),
        'std': np.std(pixels),
        'min': np.min(pixels),
        'max': np.max(pixels),
        'percentile_25': np.percentile(pixels, 25),
        'percentile_75': np.percentile(pixels, 75)
    }




def create_synthetic_dataset(output_dir: str = "synthetic_data", 
                            n_frames: int = 20, n_cells: int = 15):
    """Create synthetic microscopy images with moving cells"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    img_size = 512
    
    # Initialize cell positions and velocities
    cell_positions = np.random.rand(n_cells, 2) * (img_size - 100) + 50
    cell_velocities = (np.random.rand(n_cells, 2) - 0.5) * 3
    cell_sizes = np.random.uniform(10, 20, n_cells)
    
    print(f"Creating {n_frames} synthetic images with {n_cells} moving cells...")
    
    for frame in range(n_frames):
        # Create blank image
        img = np.zeros((img_size, img_size), dtype=np.float32)
        
        # Add cells
        for cell_idx in range(n_cells):
            x, y = cell_positions[cell_idx].astype(int)
            size = int(cell_sizes[cell_idx])
            
            # Draw circular cell with gradient
            for dy in range(-size, size + 1):
                for dx in range(-size, size + 1):
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < size:
                        px, py = x + dx, y + dy
                        if 0 <= px < img_size and 0 <= py < img_size:
                            # Gaussian-like intensity
                            intensity = 0.9 * np.exp(-(dist / size) ** 2)
                            img[py, px] = max(img[py, px], intensity)
            
            # Update position
            cell_positions[cell_idx] += cell_velocities[cell_idx]
            
            # Bounce off edges
            for dim in range(2):
                if cell_positions[cell_idx, dim] < 50:
                    cell_positions[cell_idx, dim] = 50
                    cell_velocities[cell_idx, dim] *= -1
                if cell_positions[cell_idx, dim] > img_size - 50:
                    cell_positions[cell_idx, dim] = img_size - 50
                    cell_velocities[cell_idx, dim] *= -1
        
        # Add noise
        noise = np.random.randn(img_size, img_size) * 0.03
        img = np.clip(img + noise, 0, 1)
        
        # Add background gradient
        y, x = np.ogrid[:img_size, :img_size]
        background = 0.1 * (1 - np.sqrt(((x - img_size/2)/img_size)**2 + 
                                       ((y - img_size/2)/img_size)**2))
        img += background
        img = np.clip(img, 0, 1)
        
        # Save image
        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(f"{output_dir}/frame_{frame:03d}.tif")
    
    print(f" Created synthetic dataset in {output_dir}/")