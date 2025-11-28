"""
Cell tracking algorithms
"""

import numpy as np
from typing import List, Dict, Optional
from scipy.optimize import linear_sum_assignment
from .detector import Cell


class KalmanFilter:
    """
    Simple Kalman filter for 2D position tracking
    """
    
    def __init__(self, dt: float = 1.0, process_noise: float = 1.0, 
                 measurement_noise: float = 1.0):
        """
        Initialize Kalman filter
        
        Args:
            dt: Time step
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        self.dt = dt
        
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # State covariance
        self.P = np.eye(4) * 1000  # High initial uncertainty
        
    def predict(self) -> np.ndarray:
        """
        Predict next state
        
        Returns:
            Predicted position [x, y]
        """
        # State prediction
        self.state = self.F @ self.state
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state[:2]
    
    def update(self, measurement: np.ndarray):
        """
        Update filter with new measurement
        
        Args:
            measurement: Measured position [x, y]
        """
        # Innovation
        y = measurement - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.state = self.state + K @ y
        
        # Covariance update
        self.P = (np.eye(4) - K @ self.H) @ self.P
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate"""
        return self.state[:2]
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate"""
        return self.state[2:]


class Track:
    """
    Represents a cell track across multiple frames
    """
    
    def __init__(self, track_id: int, initial_cell: Cell, frame_idx: int):
        """
        Initialize track
        
        Args:
            track_id: Unique track identifier
            initial_cell: First cell detection
            frame_idx: Frame index of first detection
        """
        self.track_id = track_id
        self.cells = [initial_cell]
        self.frame_indices = [frame_idx]
        self.kalman = KalmanFilter()
        self.kalman.state[:2] = np.array([initial_cell.x, initial_cell.y])
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        
    def predict(self) -> np.ndarray:
        """
        Predict next position
        
        Returns:
            Predicted [x, y]
        """
        self.age += 1
        self.time_since_update += 1
        return self.kalman.predict()
    
    def update(self, cell: Cell, frame_idx: int):
        """
        Update track with new detection
        
        Args:
            cell: New cell detection
            frame_idx: Frame index
        """
        self.cells.append(cell)
        self.frame_indices.append(frame_idx)
        self.kalman.update(np.array([cell.x, cell.y]))
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
    
    def mark_missed(self):
        """Mark that this track was not matched in current frame"""
        self.hit_streak = 0
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate"""
        return self.kalman.get_position()
    
    def get_trajectory(self) -> np.ndarray:
        """
        Get full trajectory
        
        Returns:
            Array of shape (n_points, 4) with [x, y, width, height]
        """
        return np.array([cell.to_array() for cell in self.cells])
    
    def is_confirmed(self, min_hits: int = 3) -> bool:
        """Check if track is confirmed (enough consecutive detections)"""
        return self.hits >= min_hits


class CellTracker:
    """
    Multi-object tracker for cells using Kalman filtering and Hungarian algorithm
    """
    
    def __init__(self, max_distance: float = 50.0, max_age: int = 5, 
                 min_hits: int = 3):
        """
        Initialize tracker
        
        Args:
            max_distance: Maximum distance for matching (pixels)
            max_age: Maximum frames to keep track without updates
            min_hits: Minimum hits to confirm track
        """
        self.max_distance = max_distance
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 0
        self.frame_count = 0
        
    def update(self, cells: List[Cell]) -> List[Track]:
        """
        Update tracker with new detections
        
        Args:
            cells: List of detected cells in current frame
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Predict positions for all tracks
        predictions = []
        for track in self.tracks:
            pred = track.predict()
            predictions.append(pred)
        
        # Match detections to tracks
        if len(self.tracks) > 0 and len(cells) > 0:
            matched, unmatched_tracks, unmatched_detections = \
                self._associate(predictions, cells)
        else:
            matched = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_detections = list(range(len(cells)))
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(cells[det_idx], self.frame_count)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_track(cells[det_idx])
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks 
                      if t.time_since_update < self.max_age]
        
        # Return confirmed tracks
        return [t for t in self.tracks if t.is_confirmed(self.min_hits)]
    
    def _associate(self, predictions: List[np.ndarray], 
                   cells: List[Cell]) -> tuple:
        """
        Associate predictions with detections using Hungarian algorithm
        
        Returns:
            matched: List of (track_idx, detection_idx) pairs
            unmatched_tracks: List of unmatched track indices
            unmatched_detections: List of unmatched detection indices
        """
        if len(predictions) == 0 or len(cells) == 0:
            return [], list(range(len(predictions))), list(range(len(cells)))
        
        # Build cost matrix (distances)
        cost_matrix = np.zeros((len(predictions), len(cells)))
        
        for i, pred in enumerate(predictions):
            for j, cell in enumerate(cells):
                dist = np.sqrt((pred[0] - cell.x)**2 + (pred[1] - cell.y)**2)
                cost_matrix[i, j] = dist
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter matches by max distance
        matched = []
        unmatched_tracks = list(range(len(predictions)))
        unmatched_detections = list(range(len(cells)))
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.max_distance:
                matched.append((i, j))
                unmatched_tracks.remove(i)
                unmatched_detections.remove(j)
        
        return matched, unmatched_tracks, unmatched_detections
    
    def _create_track(self, cell: Cell):
        """Create new track from detection"""
        track = Track(self.next_id, cell, self.frame_count)
        self.next_id += 1
        self.tracks.append(track)
    
    def get_all_trajectories(self) -> List[np.ndarray]:
        """
        Get all confirmed trajectories
        
        Returns:
            List of trajectories, each shape (n_frames, 4)
        """
        confirmed_tracks = [t for t in self.tracks 
                           if t.is_confirmed(self.min_hits)]
        return [t.get_trajectory() for t in confirmed_tracks]
