"""
Cell Morphology Analysis Module
Detailed shape and structure analysis for cell characterization
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from sklearn.decomposition import PCA


@dataclass
class MorphologyFeatures:
    """Complete set of morphological features for a cell"""
    # Basic measurements
    area: float
    perimeter: float
    
    # Shape descriptors
    circularity: float
    eccentricity: float
    solidity: float
    extent: float
    aspect_ratio: float
    
    # Orientation
    orientation: float  # Angle in degrees
    major_axis_length: float
    minor_axis_length: float
    
    # Texture features
    mean_intensity: float
    std_intensity: float
    entropy: float
    
    # Advanced shape
    compactness: float
    convexity: float
    roundness: float
    elongation: float
    
    # Hu moments (invariant features)
    hu_moment_1: float
    hu_moment_2: float
    hu_moment_3: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML"""
        return np.array(list(asdict(self).values()))


class MorphologyAnalyzer:
    """
    Detailed morphological analysis of cells
    
    Features calculated:
    - Geometric: area, perimeter, circularity, aspect ratio
    - Shape: eccentricity, solidity, extent, orientation
    - Texture: mean, std, entropy of pixel intensities
    - Advanced: Hu moments, compactness, convexity
    """
    
    def __init__(self):
        """Initialize morphology analyzer"""
        pass
    
    def extract_features(self, 
                        cell,
                        image: np.ndarray,
                        contour: Optional[np.ndarray] = None) -> MorphologyFeatures:
        """
        Extract all morphological features from a cell
        
        Args:
            cell: Cell object
            image: Original grayscale image
            contour: Optional pre-computed contour
            
        Returns:
            MorphologyFeatures object
        """
        # Create mask for cell
        mask = np.zeros(image.shape, dtype=np.uint8)
        center = (int(cell.x), int(cell.y))
        radius = int(np.sqrt(cell.area / np.pi))
        cv2.circle(mask, center, radius, 255, -1)
        
        # Find contour if not provided
        if contour is None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return self._get_default_features()
            contour = contours[0]
        
        # Basic measurements
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Prevent division by zero
        if area < 1 or perimeter < 1:
            return self._get_default_features()
        
        # Shape descriptors
        circularity = self._calculate_circularity(area, perimeter)
        
        # Fit ellipse for advanced features
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (MA, ma), angle = ellipse
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)
            
            eccentricity = np.sqrt(1 - (minor_axis / max(major_axis, 1e-10))**2)
            aspect_ratio = major_axis / max(minor_axis, 1e-10)
            orientation = angle
        else:
            major_axis = cell.width
            minor_axis = cell.height
            eccentricity = 0.5
            aspect_ratio = max(cell.width, cell.height) / max(min(cell.width, cell.height), 1)
            orientation = 0
        
        # Solidity (convex hull ratio)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1) if hull_area > 0 else 0
        
        # Extent (area / bounding box area)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        extent = area / max(bbox_area, 1)
        
        # Texture features
        cell_pixels = image[mask > 0]
        if len(cell_pixels) > 0:
            mean_intensity = np.mean(cell_pixels)
            std_intensity = np.std(cell_pixels)
            entropy = self._calculate_entropy(cell_pixels)
        else:
            mean_intensity = 0
            std_intensity = 0
            entropy = 0
        
        # Advanced shape features
        compactness = (perimeter ** 2) / max(area, 1)
        convexity = cv2.arcLength(hull, True) / max(perimeter, 1)
        roundness = (4 * area) / (np.pi * major_axis ** 2) if major_axis > 0 else 0
        elongation = 1 - (minor_axis / max(major_axis, 1e-10))
        
        # Hu moments (shape descriptors invariant to translation, rotation, scale)
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        # Use log transform to reduce range
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        return MorphologyFeatures(
            area=float(area),
            perimeter=float(perimeter),
            circularity=float(circularity),
            eccentricity=float(eccentricity),
            solidity=float(solidity),
            extent=float(extent),
            aspect_ratio=float(aspect_ratio),
            orientation=float(orientation),
            major_axis_length=float(major_axis),
            minor_axis_length=float(minor_axis),
            mean_intensity=float(mean_intensity),
            std_intensity=float(std_intensity),
            entropy=float(entropy),
            compactness=float(compactness),
            convexity=float(convexity),
            roundness=float(roundness),
            elongation=float(elongation),
            hu_moment_1=float(hu_moments[0]),
            hu_moment_2=float(hu_moments[1]),
            hu_moment_3=float(hu_moments[2])
        )
    
    def _calculate_circularity(self, area: float, perimeter: float) -> float:
        """Calculate circularity (4π*area/perimeter²)"""
        if perimeter < 1e-10:
            return 0.0
        return (4 * np.pi * area) / (perimeter ** 2)
    
    def _calculate_entropy(self, pixels: np.ndarray) -> float:
        """Calculate Shannon entropy of pixel intensities"""
        # Quantize to 256 levels
        pixels_uint8 = (pixels * 255).astype(np.uint8)
        
        # Calculate histogram
        hist, _ = np.histogram(pixels_uint8, bins=256, range=(0, 256))
        
        # Normalize to probabilities
        hist = hist / max(hist.sum(), 1)
        
        # Remove zeros
        hist = hist[hist > 0]
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return float(entropy)
    
    def _get_default_features(self) -> MorphologyFeatures:
        """Return default features when calculation fails"""
        return MorphologyFeatures(
            area=0, perimeter=0, circularity=0, eccentricity=0,
            solidity=0, extent=0, aspect_ratio=0, orientation=0,
            major_axis_length=0, minor_axis_length=0,
            mean_intensity=0, std_intensity=0, entropy=0,
            compactness=0, convexity=0, roundness=0, elongation=0,
            hu_moment_1=0, hu_moment_2=0, hu_moment_3=0
        )
    
    def classify_morphology(self, features: MorphologyFeatures) -> str:
        """
        Classify cell morphology into basic types
        
        Types:
        - Round: High circularity, low elongation
        - Elongated: Low circularity, high aspect ratio
        - Irregular: Low circularity, low solidity
        - Compact: High solidity, high roundness
        """
        if features.circularity > 0.8 and features.elongation < 0.3:
            return "round"
        elif features.aspect_ratio > 2.5 or features.elongation > 0.6:
            return "elongated"
        elif features.solidity < 0.7 or features.convexity < 0.8:
            return "irregular"
        elif features.solidity > 0.85 and features.roundness > 0.7:
            return "compact"
        else:
            return "intermediate"
    
    def analyze_population(self, 
                          cells: List,
                          image: np.ndarray) -> Dict:
        """
        Analyze morphology of entire cell population
        
        Returns:
            Statistics for each morphological feature
        """
        all_features = []
        
        for cell in cells:
            features = self.extract_features(cell, image)
            all_features.append(features)
        
        if not all_features:
            return {}
        
        # Convert to arrays for statistics
        feature_dict = all_features[0].to_dict()
        stats = {}
        
        for key in feature_dict.keys():
            values = [getattr(f, key) for f in all_features]
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }
        
        # Add morphology type distribution
        morphology_types = [self.classify_morphology(f) for f in all_features]
        type_counts = {}
        for mtype in morphology_types:
            type_counts[mtype] = type_counts.get(mtype, 0) + 1
        
        stats['morphology_distribution'] = type_counts
        
        return stats
    
    def compare_populations(self,
                           population1: List,
                           population2: List,
                           image1: np.ndarray,
                           image2: np.ndarray,
                           feature_name: str = 'circularity') -> Dict:
        """
        Compare morphological features between two populations
        """
        from scipy import stats as scipy_stats
        
        # Extract features
        features1 = [self.extract_features(c, image1) for c in population1]
        features2 = [self.extract_features(c, image2) for c in population2]
        
        # Get specific feature values
        values1 = [getattr(f, feature_name) for f in features1]
        values2 = [getattr(f, feature_name) for f in features2]
        
        # Statistical comparison
        t_stat, p_value = scipy_stats.ttest_ind(values1, values2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(values1)**2 + np.std(values2)**2) / 2)
        cohens_d = (np.mean(values1) - np.mean(values2)) / max(pooled_std, 1e-10)
        
        return {
            'feature': feature_name,
            'population1_mean': np.mean(values1),
            'population1_std': np.std(values1),
            'population2_mean': np.mean(values2),
            'population2_std': np.std(values2),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }
    
    def visualize_morphology(self,
                            cells: List,
                            image: np.ndarray,
                            feature: str = 'circularity') -> np.ndarray:
        """
        Visualize cells color-coded by morphological feature
        """
        # Create color image
        vis = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Extract feature values
        features_list = [self.extract_features(cell, image) for cell in cells]
        values = [getattr(f, feature) for f in features_list]
        
        if not values:
            return vis
        
        # Normalize values to 0-1
        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val
        
        if value_range < 1e-10:
            normalized = [0.5] * len(values)
        else:
            normalized = [(v - min_val) / value_range for v in values]
        
        # Draw cells with color map
        import matplotlib.cm as cm
        colormap = cm.get_cmap('viridis')
        
        for cell, norm_val in zip(cells, normalized):
            # Get color
            color_rgb = colormap(norm_val)[:3]
            color_bgr = tuple(int(c * 255) for c in reversed(color_rgb))
            
            # Draw
            center = (int(cell.x), int(cell.y))
            radius = int(np.sqrt(cell.area / np.pi))
            cv2.circle(vis, center, radius, color_bgr, 2)
        
        # Add colorbar legend
        legend_height = 20
        legend_width = vis.shape[1] - 100
        legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
        
        for i in range(legend_width):
            norm_pos = i / legend_width
            color_rgb = colormap(norm_pos)[:3]
            color_bgr = tuple(int(c * 255) for c in reversed(color_rgb))
            legend[:, i] = color_bgr
        
        # Add legend to image
        vis[-legend_height-40:-20, 50:50+legend_width] = legend
        
        # Add text labels
        cv2.putText(vis, f"{feature.upper()}", (50, vis.shape[0]-legend_height-45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"{min_val:.2f}", (50, vis.shape[0]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis, f"{max_val:.2f}", (50+legend_width-50, vis.shape[0]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis
    
    def export_features(self,
                       cells: List,
                       image: np.ndarray,
                       output_path: str):
        """Export morphological features to CSV"""
        import pandas as pd
        
        records = []
        
        for i, cell in enumerate(cells):
            features = self.extract_features(cell, image)
            record = features.to_dict()
            record['cell_id'] = i
            record['x'] = cell.x
            record['y'] = cell.y
            record['morphology_type'] = self.classify_morphology(features)
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"Morphology features saved to {output_path}")


def example_usage():
    """Example of morphology analysis"""
    from pipeline import Pipeline
    
    # Run detection
    pipeline = Pipeline()
    pipeline.load_images("path/to/images")
    pipeline.denoise()
    pipeline.detect_cells()
    
    # Initialize analyzer
    analyzer = MorphologyAnalyzer()
    
    # Analyze single cell
    if pipeline.detections[0]:
        cell = pipeline.detections[0][0]
        features = analyzer.extract_features(cell, pipeline.images[0])
        
        print("\nCell Morphology Features:")
        for key, value in features.to_dict().items():
            print(f"  {key}: {value:.3f}")
        
        morphology_type = analyzer.classify_morphology(features)
        print(f"\nMorphology type: {morphology_type}")
    
    # Analyze population
    pop_stats = analyzer.analyze_population(
        pipeline.detections[0],
        pipeline.images[0]
    )
    
    print("\nPopulation Statistics:")
    print(f"  Mean circularity: {pop_stats['circularity']['mean']:.3f}")
    print(f"  Mean eccentricity: {pop_stats['eccentricity']['mean']:.3f}")
    print(f"\n  Morphology distribution:")
    for mtype, count in pop_stats['morphology_distribution'].items():
        print(f"    {mtype}: {count} cells")
    
    # Visualize
    vis = analyzer.visualize_morphology(
        pipeline.detections[0],
        pipeline.images[0],
        feature='circularity'
    )
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))
    plt.imshow(vis)
    plt.title('Cells colored by Circularity')
    plt.axis('off')
    plt.savefig('morphology_visualization.png', dpi=150, bbox_inches='tight')
    
    # Export features
    analyzer.export_features(
        pipeline.detections[0],
        pipeline.images[0],
        'morphology_features.csv'
    )


if __name__ == "__main__":
    example_usage()

