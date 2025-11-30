"""
Apoptosis (Cell Death) Detection Module
Identifies programmed cell death events based on morphological changes
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import cv2


@dataclass
class ApoptosisEvent:
    """Represents a cell death event"""
    track_id: int
    death_frame: int
    death_time: float
    confidence: float
    death_type: str  # 'apoptosis', 'necrosis', 'unknown'
    
    # Apoptotic features
    shrinkage_ratio: float  # Cell size reduction
    intensity_increase: float  # Chromatin condensation
    circularity_decrease: float  # Loss of circular shape
    fragmentation_detected: bool  # Cell breaking apart
    
    def __repr__(self):
        return (f"Apoptosis(track={self.track_id}, frame={self.death_frame}, "
                f"type={self.death_type}, confidence={self.confidence:.2f})")


class ApoptosisDetector:
    """
    Detect cell death (apoptosis and necrosis) based on morphological changes
    
    Apoptosis characteristics:
    1. Cell shrinkage (decrease in area)
    2. Chromatin condensation (increased nuclear intensity)
    3. Membrane blebbing (irregular shape)
    4. Cell fragmentation (apoptotic bodies)
    5. Loss of adhesion (cell rounding)
    
    Necrosis characteristics:
    1. Cell swelling (increase in area)
    2. Loss of membrane integrity (decreased intensity)
    3. Irregular morphology
    """
    
    def __init__(self,
                 shrinkage_threshold: float = 0.7,  # 30% area reduction
                 intensity_change_threshold: float = 0.2,
                 min_track_length: int = 5,
                 min_confidence: float = 0.6):
        """
        Args:
            shrinkage_threshold: Ratio of final/initial area for apoptosis
            intensity_change_threshold: Minimum intensity change to detect
            min_track_length: Minimum track length to analyze
            min_confidence: Minimum confidence to report death
        """
        self.shrinkage_threshold = shrinkage_threshold
        self.intensity_change_threshold = intensity_change_threshold
        self.min_track_length = min_track_length
        self.min_confidence = min_confidence
    
    def detect_apoptosis(self, 
                         tracks: List,
                         images: List[np.ndarray]) -> List[ApoptosisEvent]:
        """
        Detect all apoptosis events in tracked cells
        
        Args:
            tracks: List of Track objects
            images: List of original images
            
        Returns:
            List of detected apoptosis events
        """
        apoptosis_events = []
        
        for track in tracks:
            # Skip short tracks
            if len(track.cells) < self.min_track_length:
                continue
            
            # Analyze track for apoptosis
            event = self._analyze_track_for_death(track, images)
            
            if event and event.confidence >= self.min_confidence:
                apoptosis_events.append(event)
        
        return apoptosis_events
    
    def _analyze_track_for_death(self,
                                  track,
                                  images: List[np.ndarray]) -> Optional[ApoptosisEvent]:
        """
        Analyze a single track for cell death
        """
        # Extract time series of features
        areas = [cell.area for cell in track.cells]
        intensities = [cell.mean_intensity for cell in track.cells]
        
        # Calculate circularities
        circularities = []
        for cell in track.cells:
            perimeter = 2 * np.pi * np.sqrt(cell.area / np.pi)
            if perimeter > 0:
                circularity = 4 * np.pi * cell.area / (perimeter ** 2)
            else:
                circularity = 0
            circularities.append(circularity)
        
        # Detect trends
        shrinkage_ratio = self._calculate_shrinkage(areas)
        intensity_change = self._calculate_intensity_change(intensities)
        circularity_change = self._calculate_circularity_loss(circularities)
        
        # Check for fragmentation
        fragmentation = self._detect_fragmentation(track, images)
        
        # Determine death type and confidence
        death_type, confidence = self._classify_death_type(
            shrinkage_ratio,
            intensity_change,
            circularity_change,
            fragmentation
        )
        
        if death_type == 'none':
            return None
        
        # Estimate death frame (when changes became significant)
        death_frame = self._estimate_death_frame(track, areas, intensities)
        
        event = ApoptosisEvent(
            track_id=track.track_id,
            death_frame=death_frame,
            death_time=death_frame,
            confidence=confidence,
            death_type=death_type,
            shrinkage_ratio=shrinkage_ratio,
            intensity_increase=intensity_change,
            circularity_decrease=circularity_change,
            fragmentation_detected=fragmentation
        )
        
        return event
    
    def _calculate_shrinkage(self, areas: List[float]) -> float:
        """
        Calculate cell shrinkage ratio
        Returns ratio of final to initial area (< 1 means shrinkage)
        """
        if len(areas) < 3:
            return 1.0
        
        # Compare last 20% to first 20%
        n = len(areas)
        window = max(2, n // 5)
        
        initial_area = np.mean(areas[:window])
        final_area = np.mean(areas[-window:])
        
        if initial_area < 1:
            return 1.0
        
        return final_area / initial_area
    
    def _calculate_intensity_change(self, intensities: List[float]) -> float:
        """
        Calculate intensity change (positive = increase)
        """
        if len(intensities) < 3:
            return 0.0
        
        n = len(intensities)
        window = max(2, n // 5)
        
        initial_intensity = np.mean(intensities[:window])
        final_intensity = np.mean(intensities[-window:])
        
        if initial_intensity < 1e-6:
            return 0.0
        
        # Normalized change
        change = (final_intensity - initial_intensity) / initial_intensity
        return change
    
    def _calculate_circularity_loss(self, circularities: List[float]) -> float:
        """
        Calculate loss of circularity (positive = more irregular)
        """
        if len(circularities) < 3:
            return 0.0
        
        n = len(circularities)
        window = max(2, n // 5)
        
        initial_circ = np.mean(circularities[:window])
        final_circ = np.mean(circularities[-window:])
        
        # Loss of circularity
        loss = initial_circ - final_circ
        return max(0, loss)
    
    def _detect_fragmentation(self,
                              track,
                              images: List[np.ndarray]) -> bool:
        """
        Detect if cell fragments into apoptotic bodies
        Look for sudden appearance of multiple small objects near cell
        """
        # This is a simplified version
        # In real implementation, would analyze nearby detections
        
        # Check if track ends abruptly (suggests fragmentation)
        if len(track.frame_indices) > 0:
            last_frame = track.frame_indices[-1]
            max_frame = len(images) - 1
            
            # If track ends before last frame, might indicate fragmentation
            if last_frame < max_frame - 3:
                return True
        
        return False
    
    def _classify_death_type(self,
                             shrinkage_ratio: float,
                             intensity_change: float,
                             circularity_change: float,
                             fragmentation: bool) -> Tuple[str, float]:
        """
        Classify type of cell death and calculate confidence
        
        Returns:
            (death_type, confidence)
        """
        features = []
        
        # Apoptosis indicators
        if shrinkage_ratio < self.shrinkage_threshold:
            features.append(('shrinkage', 0.3))
        
        if intensity_change > self.intensity_change_threshold:
            features.append(('intensity', 0.25))
        
        if circularity_change > 0.1:
            features.append(('shape', 0.2))
        
        if fragmentation:
            features.append(('fragmentation', 0.25))
        
        # Calculate confidence
        if len(features) == 0:
            return 'none', 0.0
        
        confidence = sum(score for _, score in features)
        
        # Classify death type
        if shrinkage_ratio < self.shrinkage_threshold and intensity_change > 0:
            death_type = 'apoptosis'
        elif shrinkage_ratio > 1.2 and intensity_change < 0:
            death_type = 'necrosis'
        else:
            death_type = 'unknown'
        
        return death_type, min(confidence, 1.0)
    
    def _estimate_death_frame(self,
                              track,
                              areas: List[float],
                              intensities: List[float]) -> int:
        """
        Estimate when cell death occurred
        """
        if len(areas) < 3:
            return track.frame_indices[-1]
        
        # Find frame where area starts decreasing significantly
        area_diff = np.diff(areas)
        
        # Find sustained decrease
        for i in range(len(area_diff) - 2):
            if np.mean(area_diff[i:i+3]) < -5:  # Sustained decrease
                return track.frame_indices[i]
        
        # Default to last frame
        return track.frame_indices[-1]
    
    def calculate_survival_statistics(self,
                                     tracks: List,
                                     apoptosis_events: List[ApoptosisEvent]) -> Dict:
        """
        Calculate survival statistics
        """
        total_cells = len(tracks)
        dead_cells = len(apoptosis_events)
        alive_cells = total_cells - dead_cells
        
        # Survival rate
        survival_rate = alive_cells / total_cells if total_cells > 0 else 0
        
        # Mean time to death
        if dead_cells > 0:
            death_times = [event.death_frame for event in apoptosis_events]
            mean_death_time = np.mean(death_times)
            median_death_time = np.median(death_times)
        else:
            mean_death_time = None
            median_death_time = None
        
        # Death types
        death_types = {}
        for event in apoptosis_events:
            death_types[event.death_type] = death_types.get(event.death_type, 0) + 1
        
        return {
            'total_cells': total_cells,
            'alive_cells': alive_cells,
            'dead_cells': dead_cells,
            'survival_rate': survival_rate,
            'mean_death_time': mean_death_time,
            'median_death_time': median_death_time,
            'death_types': death_types
        }
    
    def visualize_apoptosis(self,
                           image: np.ndarray,
                           apoptosis_events: List[ApoptosisEvent],
                           all_tracks: List,
                           frame_idx: int) -> np.ndarray:
        """
        Visualize apoptosis events
        """
        # Convert to color
        if len(image.shape) == 2:
            vis = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Draw apoptotic cells
        for event in apoptosis_events:
            if event.death_frame != frame_idx:
                continue
            
            # Find track
            track = next((t for t in all_tracks if t.track_id == event.track_id), None)
            if not track:
                continue
            
            # Get cell at death frame
            try:
                idx = track.frame_indices.index(frame_idx)
                cell = track.cells[idx]
            except ValueError:
                continue
            
            # Draw red circle for dead cell
            cv2.circle(vis, (int(cell.x), int(cell.y)), 20, (0, 0, 255), 3)
            
            # Draw skull emoji or text
            label = f"DEAD #{event.track_id}"
            cv2.putText(vis, label,
                       (int(cell.x) - 30, int(cell.y) - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Add death type
            cv2.putText(vis, event.death_type,
                       (int(cell.x) - 30, int(cell.y) + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis


def example_usage():
    """Example of using the apoptosis detector"""
    from pipeline import Pipeline
    
    # Run normal analysis
    pipeline = Pipeline()
    pipeline.load_images("path/to/images")
    pipeline.denoise()
    pipeline.detect_cells()
    trajectory = pipeline.track_cells()
    
    # Detect apoptosis
    apoptosis_detector = ApoptosisDetector(
        shrinkage_threshold=0.7,
        intensity_change_threshold=0.2,
        min_confidence=0.6
    )
    
    apoptosis_events = apoptosis_detector.detect_apoptosis(
        trajectory.tracks,
        pipeline.images
    )
    
    print(f"\nDetected {len(apoptosis_events)} apoptosis events:")
    for event in apoptosis_events:
        print(f"  {event}")
    
    # Calculate survival statistics
    stats = apoptosis_detector.calculate_survival_statistics(
        trajectory.tracks,
        apoptosis_events
    )
    
    print(f"\nSurvival Statistics:")
    print(f"  Total cells: {stats['total_cells']}")
    print(f"  Alive: {stats['alive_cells']}")
    print(f"  Dead: {stats['dead_cells']}")
    print(f"  Survival rate: {stats['survival_rate']:.1%}")
    if stats['mean_death_time']:
        print(f"  Mean time to death: {stats['mean_death_time']:.1f} frames")
    
    # Export to CSV
    import pandas as pd
    
    events_data = [{
        'track_id': e.track_id,
        'death_frame': e.death_frame,
        'death_type': e.death_type,
        'confidence': e.confidence,
        'shrinkage_ratio': e.shrinkage_ratio,
        'intensity_increase': e.intensity_increase,
        'fragmentation': e.fragmentation_detected
    } for e in apoptosis_events]
    
    df = pd.DataFrame(events_data)
    df.to_csv('apoptosis_events.csv', index=False)
    print("\nApoptosis data saved to 'apoptosis_events.csv'")


if __name__ == "__main__":
    example_usage()

