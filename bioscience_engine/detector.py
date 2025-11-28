"""
Fixed cell detection algorithms using OpenCV
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Cell:
    """Data class representing a detected cell"""
    x: float
    y: float
    width: float
    height: float
    area: float
    confidence: float
    mean_intensity: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.width, self.height])
    
    def get_bbox(self) -> Tuple[int, int, int, int]:
        x1 = int(self.x - self.width / 2)
        y1 = int(self.y - self.height / 2)
        x2 = int(self.x + self.width / 2)
        y2 = int(self.y + self.height / 2)
        return (x1, y1, x2, y2)


class CellDetector:
    """
    Fixed cell detection using multiple computer vision techniques
    """
    
    def __init__(self, min_size: int = 50, max_size: int = 500, 
                 threshold: float = 0.7):
        self.min_size = min_size
        self.max_size = max_size
        self.threshold = threshold
        
    def detect(self, image: np.ndarray, method: str = 'blob') -> List[Cell]:
        """Detect cells in image"""
        if method == 'blob':
            return self._detect_blob(image)
        elif method == 'watershed':
            return self._detect_watershed(image)
        elif method == 'contour':
            return self._detect_contour(image)
        elif method == 'hybrid':
            return self._detect_hybrid(image)
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better cell detection
        
        This is the KEY FIX - proper preprocessing for bright cells on dark background
        """
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply strong denoising
        denoised = cv2.GaussianBlur(img_uint8, (5, 5), 1.5)
        
        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def _detect_blob(self, image: np.ndarray) -> List[Cell]:
        """
        Detect cells using blob detection - FIXED VERSION
        """
        # Preprocess image
        img_preprocessed = self._preprocess_image(image)
        
        # Setup SimpleBlobDetector parameters - FIXED for bright blobs
        params = cv2.SimpleBlobDetector_Params()
        
        # Detect BRIGHT blobs (not dark ones!)
        params.filterByColor = True
        params.blobColor = 255  # Look for WHITE/BRIGHT blobs
        
        # Filter by area
        params.filterByArea = True
        params.minArea = self.min_size
        params.maxArea = self.max_size
        
        # Filter by circularity (cells are roughly circular)
        params.filterByCircularity = True
        params.minCircularity = 0.4  # Slightly more strict
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.6  # More strict
        
        # Filter by inertia (roundness)
        params.filterByInertia = True
        params.minInertiaRatio = 0.3  # More strict
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(img_preprocessed)
        
        # Convert to Cell objects
        cells = []
        for kp in keypoints:
            size = kp.size
            
            cell = Cell(
                x=kp.pt[0],
                y=kp.pt[1],
                width=size,
                height=size,
                area=np.pi * (size / 2) ** 2,
                confidence=kp.response if kp.response > 0 else 0.8
            )
            
            # Calculate mean intensity
            x1, y1, x2, y2 = cell.get_bbox()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                cell.mean_intensity = np.mean(image[y1:y2, x1:x2])
            
            if cell.confidence >= self.threshold:
                cells.append(cell)
        
        return cells
    
    def _detect_watershed(self, image: np.ndarray) -> List[Cell]:
        """
        Detect cells using watershed segmentation - FIXED VERSION
        """
        # Preprocess
        img_preprocessed = self._preprocess_image(image)
        
        # Apply Otsu thresholding to find bright regions
        _, binary = cv2.threshold(img_preprocessed, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Additional check - if most pixels are white, we need to invert
        # This handles the case where background is detected instead of cells
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        # Clean up with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area using distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 
                                   255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        img_color = cv2.cvtColor(img_preprocessed, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        # Extract cells from markers
        cells = []
        for label in range(2, markers.max() + 1):
            mask = (markers == label).astype(np.uint8)
            
            # Find contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
            
            contour = contours[0]
            area = cv2.contourArea(contour)
            
            # Filter by size
            if area < self.min_size or area > self.max_size:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx = x + w / 2
                cy = y + h / 2
            
            cell = Cell(
                x=cx,
                y=cy,
                width=w,
                height=h,
                area=area,
                confidence=0.85,
                mean_intensity=np.mean(image[mask > 0])
            )
            
            cells.append(cell)
        
        return cells
    
    def _detect_contour(self, image: np.ndarray) -> List[Cell]:
        """
        Detect cells using contour detection - FIXED VERSION
        """
        # Preprocess
        img_preprocessed = self._preprocess_image(image)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            img_preprocessed, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 
            blockSize=15,  # Larger block size
            C=-2  # Negative C to detect bright regions
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size
            if area < self.min_size or area > self.max_size:
                continue
            
            # Get bounding box and centroid
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx = x + w / 2
                cy = y + h / 2
            
            # Calculate confidence based on circularity and aspect ratio
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                # Penalize elongated shapes
                confidence = min(circularity * 1.2, 1.0) * (1.0 / max(aspect_ratio, 1.0))
            else:
                confidence = 0.5
            
            if confidence < self.threshold:
                continue
            
            # Create mask for intensity calculation
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            cell = Cell(
                x=cx,
                y=cy,
                width=w,
                height=h,
                area=area,
                confidence=confidence,
                mean_intensity=np.mean(image[mask > 0])
            )
            
            cells.append(cell)
        
        return cells
    
    def _detect_hybrid(self, image: np.ndarray) -> List[Cell]:
        """
        Detect cells using hybrid approach - FIXED VERSION
        """
        # Get detections from multiple methods
        blob_cells = self._detect_blob(image)
        watershed_cells = self._detect_watershed(image)
        
        # Merge detections
        all_cells = blob_cells + watershed_cells
        
        if len(all_cells) == 0:
            return []
        
        # Use non-maximum suppression to remove duplicates
        merged_cells = self._non_max_suppression(all_cells, overlap_threshold=0.3)
        
        return merged_cells
    
    def _non_max_suppression(self, cells: List[Cell], 
                            overlap_threshold: float = 0.3) -> List[Cell]:
        """Remove duplicate detections using non-maximum suppression"""
        if len(cells) == 0:
            return []
        
        # Sort by confidence
        cells = sorted(cells, key=lambda c: c.confidence, reverse=True)
        
        keep = []
        
        while len(cells) > 0:
            current = cells[0]
            keep.append(current)
            cells = cells[1:]
            
            filtered = []
            for cell in cells:
                if self._calculate_iou(current, cell) < overlap_threshold:
                    filtered.append(cell)
            
            cells = filtered
        
        return keep
    
    def _calculate_iou(self, cell1: Cell, cell2: Cell) -> float:
        """Calculate Intersection over Union between two cells"""
        x1_1, y1_1, x2_1, y2_1 = cell1.get_bbox()
        x1_2, y1_2, x2_2, y2_2 = cell2.get_bbox()
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
        