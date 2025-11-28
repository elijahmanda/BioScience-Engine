"""
Cell Division Detection Module
Identifies mitosis events and tracks parent-daughter relationships
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from PIL import Image

from bioscience_engine.utils import create_synthetic_dataset




@dataclass
class DivisionEvent:
    """Represents a cell division event"""
    parent_track_id: int
    daughter_track_ids: Tuple[int, int]
    division_frame: int
    division_time: float
    parent_area: float
    daughters_area: Tuple[float, float]
    confidence: float
    division_angle: float  # Angle between daughter cells
    
    def __repr__(self):
        return (f"Division(parent={self.parent_track_id}, "
                f"daughters={self.daughter_track_ids}, "
                f"frame={self.division_frame}, confidence={self.confidence:.2f})")


class CellDivisionDetector:
    """
    Detect cell division events in tracked trajectories
    
    Division is characterized by:
    1. Sudden increase in area/size
    2. Cell elongation before division
    3. Splitting into two daughter cells
    4. Two new tracks appearing near parent's last position
    """
    
    def __init__(self, 
                 area_increase_threshold: float = 1.5,
                 max_distance_threshold: float = 50.0,
                 min_confidence: float = 0.7,
                 elongation_threshold: float = 1.3):
        """
        Args:
            area_increase_threshold: Minimum area increase ratio to flag division
            max_distance_threshold: Max distance daughters can be from parent
            min_confidence: Minimum confidence to report division
            elongation_threshold: Aspect ratio threshold for elongation
        """
        self.area_increase_threshold = area_increase_threshold
        self.max_distance_threshold = max_distance_threshold
        self.min_confidence = min_confidence
        self.elongation_threshold = elongation_threshold
        
    def detect_divisions(self, tracks: List, frame_detections: List[List]) -> List[DivisionEvent]:
        """
        Detect all division events in a set of tracks
        
        Args:
            tracks: List of Track objects
            frame_detections: List of cell detections per frame (List[List[Cell]])
            
        Returns:
            List of detected division events
        """
        divisions = []
        
        # Build track lookup by frame
        track_by_frame = self._build_track_index(tracks)
        
        # Check each track for division signatures
        for track in tracks:
            # Only check tracks that end before the last frame
            if track.frame_indices[-1] < len(frame_detections) - 1:
                division = self._check_track_for_division(
                    track, tracks, frame_detections, track_by_frame
                )
                if division and division.confidence >= self.min_confidence:
                    divisions.append(division)
        
        return divisions
    
    def _build_track_index(self, tracks: List) -> Dict[int, List]:
        """Build index of which tracks exist in each frame"""
        track_by_frame = {}
        for track in tracks:
            for frame_idx in track.frame_indices:
                if frame_idx not in track_by_frame:
                    track_by_frame[frame_idx] = []
                track_by_frame[frame_idx].append(track)
        return track_by_frame
    
    def _check_track_for_division(self, 
                                   parent_track, 
                                   all_tracks: List,
                                   frame_detections: List[List],
                                   track_by_frame: Dict) -> Optional[DivisionEvent]:
        """
        Check if a track ends due to cell division
        """
        # Get last few detections of parent
        if len(parent_track.cells) < 3:
            return None
        
        last_frame = parent_track.frame_indices[-1]
        last_cell = parent_track.cells[-1]
        
        # Check for area increase trend (pre-division growth)
        area_trend = self._check_area_increase(parent_track)
        
        # Check for elongation (cells elongate before splitting)
        elongation = self._check_elongation(parent_track)
        
        # Look for daughter cells in next few frames
        daughter_candidates = self._find_daughter_candidates(
            last_cell, last_frame, all_tracks, track_by_frame
        )
        
        if len(daughter_candidates) < 2:
            return None
        
        # Find best pair of daughter cells
        best_pair, pair_confidence = self._find_best_daughter_pair(
            last_cell, daughter_candidates
        )
        
        if best_pair is None:
            return None
        
        daughter1, daughter2 = best_pair
        
        # Calculate overall confidence
        confidence = self._calculate_division_confidence(
            area_trend, elongation, pair_confidence
        )
        
        # Calculate division angle
        angle = self._calculate_division_angle(last_cell, daughter1, daughter2)
        
        # Create division event
        division = DivisionEvent(
            parent_track_id=parent_track.track_id,
            daughter_track_ids=(daughter1.track_id, daughter2.track_id),
            division_frame=last_frame,
            division_time=last_frame,  # Could be actual time if available
            parent_area=last_cell.area,
            daughters_area=(daughter1.cells[0].area, daughter2.cells[0].area),
            confidence=confidence,
            division_angle=angle
        )
        
        return division
    
    def _check_area_increase(self, track) -> float:
        """
        Check if cell area increased before division (0-1)
        """
        if len(track.cells) < 3:
            return 0.0
        
        # Compare last cells to earlier cells
        recent_areas = [c.area for c in track.cells[-3:]]
        earlier_areas = [c.area for c in track.cells[:min(5, len(track.cells)-3)]]
        
        if not earlier_areas:
            return 0.0
        
        avg_recent = np.mean(recent_areas)
        avg_earlier = np.mean(earlier_areas)
        
        if avg_earlier < 1:
            return 0.0
        
        area_increase = avg_recent / avg_earlier
        
        # Normalize to 0-1
        if area_increase >= self.area_increase_threshold:
            return min(1.0, (area_increase - 1.0) / self.area_increase_threshold)
        else:
            return 0.0
    
    def _check_elongation(self, track) -> float:
        """
        Check if cell became elongated before division (0-1)
        """
        if len(track.cells) < 2:
            return 0.0
        
        # Check last few cells for elongation
        recent_cells = track.cells[-3:]
        
        elongations = []
        for cell in recent_cells:
            aspect_ratio = max(cell.width, cell.height) / max(min(cell.width, cell.height), 1)
            elongations.append(aspect_ratio)
        
        max_elongation = max(elongations)
        
        # Normalize to 0-1
        if max_elongation >= self.elongation_threshold:
            return min(1.0, (max_elongation - 1.0) / self.elongation_threshold)
        else:
            return 0.0
    
    def _find_daughter_candidates(self, 
                                   parent_cell, 
                                   last_frame: int,
                                   all_tracks: List,
                                   track_by_frame: Dict) -> List:
        """
        Find tracks that start near the parent's last position
        """
        candidates = []
        
        # Look in next few frames
        search_frames = range(last_frame + 1, min(last_frame + 4, max(track_by_frame.keys()) + 1))
        
        for frame_idx in search_frames:
            if frame_idx not in track_by_frame:
                continue
            
            for track in track_by_frame[frame_idx]:
                # Must be a new track (starts in this frame or shortly before)
                if track.frame_indices[0] < last_frame:
                    continue
                
                # Must start near parent's last position
                first_cell = track.cells[0]
                distance = np.sqrt(
                    (first_cell.x - parent_cell.x)**2 + 
                    (first_cell.y - parent_cell.y)**2
                )
                
                if distance <= self.max_distance_threshold:
                    candidates.append(track)
        
        return candidates
    
    def _find_best_daughter_pair(self, 
                                  parent_cell,
                                  candidates: List) -> Tuple[Optional[Tuple], float]:
        """
        Find the best pair of daughter cells from candidates
        """
        if len(candidates) < 2:
            return None, 0.0
        
        best_pair = None
        best_score = 0.0
        
        # Try all pairs
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                daughter1 = candidates[i]
                daughter2 = candidates[j]
                
                score = self._score_daughter_pair(parent_cell, daughter1, daughter2)
                
                if score > best_score:
                    best_score = score
                    best_pair = (daughter1, daughter2)
        
        return best_pair, best_score
    
    def _score_daughter_pair(self, parent_cell, daughter1, daughter2) -> float:
        """
        Score how likely two tracks are daughter cells
        
        Good daughter pairs should:
        1. Be similar in size
        2. Be on opposite sides of parent
        3. Have combined area ≈ parent area
        4. Be roughly equidistant from parent
        """
        cell1 = daughter1.cells[0]
        cell2 = daughter2.cells[0]
        
        # Size similarity (0-1)
        size_ratio = min(cell1.area, cell2.area) / max(cell1.area, cell2.area, 1)
        
        # Area conservation (daughters' total ≈ parent's)
        total_daughter_area = cell1.area + cell2.area
        area_ratio = min(parent_cell.area, total_daughter_area) / max(parent_cell.area, total_daughter_area, 1)
        area_score = area_ratio if 0.8 <= area_ratio <= 1.5 else 0.0
        
        # Spatial arrangement (daughters on opposite sides)
        vec1 = np.array([cell1.x - parent_cell.x, cell1.y - parent_cell.y])
        vec2 = np.array([cell2.x - parent_cell.x, cell2.y - parent_cell.y])
        
        # Angle between vectors (should be close to 180 degrees)
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms > 0:
            cos_angle = dot_product / norms
            # -1 = 180 degrees (perfect), 1 = 0 degrees (bad)
            spatial_score = (1 - cos_angle) / 2  # Convert to 0-1 (1 is best)
        else:
            spatial_score = 0.0
        
        # Distance balance (should be roughly equidistant)
        dist1 = np.linalg.norm(vec1)
        dist2 = np.linalg.norm(vec2)
        distance_balance = min(dist1, dist2) / max(dist1, dist2, 1)
        
        # Combined score
        score = (size_ratio * 0.3 + 
                area_score * 0.3 + 
                spatial_score * 0.25 + 
                distance_balance * 0.15)
        
        return score
    
    def _calculate_division_confidence(self, 
                                       area_trend: float,
                                       elongation: float,
                                       pair_confidence: float) -> float:
        """
        Calculate overall confidence in division detection
        """
        # Weighted combination of features
        confidence = (pair_confidence * 0.6 + 
                     area_trend * 0.2 + 
                     elongation * 0.2)
        return confidence
    
    def _calculate_division_angle(self, parent_cell, daughter1, daughter2) -> float:
        """
        Calculate angle of division axis (in degrees)
        """
        cell1 = daughter1.cells[0]
        cell2 = daughter2.cells[0]
        
        # Vector from daughter1 to daughter2
        dx = cell2.x - cell1.x
        dy = cell2.y - cell1.y
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return angle
    
    def visualize_divisions(self, 
                           image: np.ndarray,
                           divisions: List[DivisionEvent],
                           all_tracks: List) -> np.ndarray:
        """
        Create visualization of detected divisions
        """
        import cv2
        
        # Convert to color if grayscale
        if len(image.shape) == 2:
            vis = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Draw each division
        for division in divisions:
            # Find tracks
            parent = next(t for t in all_tracks if t.track_id == division.parent_track_id)
            daughter1 = next(t for t in all_tracks if t.track_id == division.daughter_track_ids[0])
            daughter2 = next(t for t in all_tracks if t.track_id == division.daughter_track_ids[1])
            
            # Get cells at division time
            parent_cell = parent.cells[-1]
            d1_cell = daughter1.cells[0]
            d2_cell = daughter2.cells[0]
            
            # Draw parent (yellow circle)
            cv2.circle(vis, (int(parent_cell.x), int(parent_cell.y)), 
                      15, (0, 255, 255), 2)
            
            # Draw daughters (green circles)
            cv2.circle(vis, (int(d1_cell.x), int(d1_cell.y)), 
                      10, (0, 255, 0), 2)
            cv2.circle(vis, (int(d2_cell.x), int(d2_cell.y)), 
                      10, (0, 255, 0), 2)
            
            # Draw lines connecting parent to daughters
            cv2.line(vis, 
                    (int(parent_cell.x), int(parent_cell.y)),
                    (int(d1_cell.x), int(d1_cell.y)),
                    (255, 0, 0), 2)
            cv2.line(vis, 
                    (int(parent_cell.x), int(parent_cell.y)),
                    (int(d2_cell.x), int(d2_cell.y)),
                    (255, 0, 0), 2)
            
            # Add text label
            label = f"Div {division.parent_track_id}"
            cv2.putText(vis, label,
                       (int(parent_cell.x) - 20, int(parent_cell.y) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis


# Example usage function
def example_usage():
    """
    Example of how to use the division detector
    """
    from bioscience_engine.pipeline import Pipeline
    
    # Run normal analysis
    pipeline = Pipeline()
    # Create synthetic data
    print("\n1  Creating synthetic dataset...")
    create_synthetic_dataset(n_frames=30, n_cells=20)
    
    # Initialize pipeline
    print("\n2  Initializing pipeline...")
    pipeline = Pipeline()
    
    # Load images
    print("\n3  Loading images...")
    pipeline.load_images("synthetic_data/")
    pipeline.denoise()
    pipeline.detect_cells()
    trajectory = pipeline.track_cells()
    
    # Detect divisions
    division_detector = CellDivisionDetector(
        area_increase_threshold=1.4,
        max_distance_threshold=60.0,
        min_confidence=0.6
    )
    
    divisions = division_detector.detect_divisions(
        trajectory.tracks,
        pipeline.detections
    )
    
    print(f"\nDetected {len(divisions)} cell division events:")
    for div in divisions:
        print(f"  {div}")
        print(f"    Confidence: {div.confidence:.2f}")
        print(f"    Division angle: {div.division_angle:.1f}°")
        print(f"    Area conservation: {sum(div.daughters_area) / div.parent_area:.2f}x")
    
    # Visualize
    frame_idx = divisions[0].division_frame if divisions else 0
    vis = division_detector.visualize_divisions(
        pipeline.images[frame_idx],
        divisions,
        trajectory.tracks
    )
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))
    plt.imshow(vis)
    plt.title(f"Detected {len(divisions)} Division Events")
    plt.axis('off')
    plt.savefig('divisions_detected.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Export division data
    import pandas as pd
    division_data = [{
        'parent_track': d.parent_track_id,
        'daughter1_track': d.daughter_track_ids[0],
        'daughter2_track': d.daughter_track_ids[1],
        'frame': d.division_frame,
        'confidence': d.confidence,
        'angle': d.division_angle,
        'parent_area': d.parent_area,
        'daughter1_area': d.daughters_area[0],
        'daughter2_area': d.daughters_area[1]
    } for d in divisions]
    
    df = pd.DataFrame(division_data)
    df.to_csv('division_events.csv', index=False)
    print("\nDivision data saved to 'division_events.csv'")


if __name__ == "__main__":
    example_usage()
