"""
Cell tracking algorithms
"""

import numpy as np
from typing import List, Dict, Optional
from scipy.optimize import linear_sum_assignment
from .detector import Cell


class KalmanFilter:
    """Simple Kalman filter for 2D position tracking"""
    
    def __init__(self, dt: float = 1.0, process_noise: float = 1.0, 
                 measurement_noise: float = 1.0):
        self.dt = dt
        
        self.state = np.zeros(4)
        
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4) * 1000
        
    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]
    
    def update(self, measurement: np.ndarray):
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
    
    def get_position(self) -> np.ndarray:
        return self.state[:2]
    
    def get_velocity(self) -> np.ndarray:
        return self.state[2:]


class Track:
    """Represents a cell track across multiple frames"""
    
    def __init__(self, track_id: int, initial_cell: Cell, frame_idx: int):
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
        self.age += 1
        self.time_since_update += 1
        return self.kalman.predict()
    
    def update(self, cell: Cell, frame_idx: int):
        self.cells.append(cell)
        self.frame_indices.append(frame_idx)
        self.kalman.update(np.array([cell.x, cell.y]))
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
    
    def mark_missed(self):
        self.hit_streak = 0
    
    def get_position(self) -> np.ndarray:
        return self.kalman.get_position()
    
    def get_trajectory(self) -> np.ndarray:
        return np.array([cell.to_array() for cell in self.cells])
    
    def is_confirmed(self, min_hits: int = 3) -> bool:
        return self.hits >= min_hits


class CellTracker:
    """Multi-object tracker for cells using Kalman filtering and Hungarian algorithm"""
    
    def __init__(self, max_distance: float = 50.0, max_age: int = 5, 
                 min_hits: int = 3):
        self.max_distance = max_distance
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 0
        self.frame_count = 0
        
    def update(self, cells: List[Cell]) -> List[Track]:
        self.frame_count += 1
        
        predictions = []
        for track in self.tracks:
            pred = track.predict()
            predictions.append(pred)
        
        if len(self.tracks) > 0 and len(cells) > 0:
            matched, unmatched_tracks, unmatched_detections = \
                self._associate(predictions, cells)
        else:
            matched = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_detections = list(range(len(cells)))
        
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(cells[det_idx], self.frame_count)
        
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        for det_idx in unmatched_detections:
            self._create_track(cells[det_idx])
        
        self.tracks = [t for t in self.tracks 
                      if t.time_since_update < self.max_age]
        
        return [t for t in self.tracks if t.is_confirmed(self.min_hits)]
    
    def _associate(self, predictions: List[np.ndarray], 
                   cells: List[Cell]) -> tuple:
        if len(predictions) == 0 or len(cells) == 0:
            return [], list(range(len(predictions))), list(range(len(cells)))
        
        cost_matrix = np.zeros((len(predictions), len(cells)))
        
        for i, pred in enumerate(predictions):
            for j, cell in enumerate(cells):
                dist = np.sqrt((pred[0] - cell.x)**2 + (pred[1] - cell.y)**2)
                cost_matrix[i, j] = dist
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
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
        track = Track(self.next_id, cell, self.frame_count)
        self.next_id += 1
        self.tracks.append(track)
    
    def get_all_trajectories(self) -> List[np.ndarray]:
        confirmed_tracks = [t for t in self.tracks 
                           if t.is_confirmed(self.min_hits)]
        return [t.get_trajectory() for t in confirmed_tracks]
    