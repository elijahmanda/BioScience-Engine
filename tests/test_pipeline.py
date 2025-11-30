"""
Unit tests for pipeline
"""

import pytest
import numpy as np
import bioscience_engine as bio
from pathlib import Path


def test_pipeline_creation():
    """Test pipeline initialization"""
    pipeline = bio.Pipeline()
    assert pipeline is not None
    assert len(pipeline.images) == 0


def test_parameter_setting():
    """Test parameter configuration"""
    pipeline = bio.Pipeline()
    pipeline.set_parameters(
        min_cell_size=100,
        detection_threshold=0.9,
        detection_method='blob'
    )
    assert pipeline.params['min_cell_size'] == 100
    assert pipeline.params['detection_threshold'] == 0.9
    assert pipeline.params['detection_method'] == 'blob'


def test_cell_detector():
    """Test cell detection"""
    # Create synthetic image with cells
    img = np.zeros((512, 512), dtype=np.float32)
    
    # Add 5 cells
    for i in range(5):
        x, y = 100 + i * 80, 256
        for dy in range(-15, 16):
            for dx in range(-15, 16):
                if dx*dx + dy*dy < 225:
                    if 0 <= y+dy < 512 and 0 <= x+dx < 512:
                        img[y+dy, x+dx] = 0.8
    
    detector = bio.CellDetector(min_size=100, max_size=1000, threshold=0.5)
    cells = detector.detect(img, method='contour')
    
    assert len(cells) > 0
    assert all(isinstance(cell, bio.detector.Cell) for cell in cells)


def test_tracking():
    """Test cell tracking"""
    tracker = bio.CellTracker(max_distance=50.0)
    
    # Create synthetic detections
    from bioscience_engine.detector import Cell
    
    # Frame 1: 3 cells
    cells1 = [
        Cell(100, 100, 20, 20, 400, 0.9),
        Cell(200, 200, 20, 20, 400, 0.9),
        Cell(300, 300, 20, 20, 400, 0.9)
    ]
    
    # Frame 2: cells moved slightly
    cells2 = [
        Cell(105, 102, 20, 20, 400, 0.9),
        Cell(198, 205, 20, 20, 400, 0.9),
        Cell(305, 298, 20, 20, 400, 0.9)
    ]
    
    tracks1 = tracker.update(cells1)
    tracks2 = tracker.update(cells2)
    
    assert len(tracker.tracks) == 3

