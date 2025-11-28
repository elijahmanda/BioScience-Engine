"""
BioScience Engine - Pure Python Implementation
High-performance computational microscopy with OpenCV
"""

from .pipeline import Pipeline, Trajectory
from .detector import CellDetector
from .tracker import CellTracker

__version__ = "0.2.0"
__all__ = ['Pipeline', 'Trajectory', 'CellDetector', 'CellTracker']