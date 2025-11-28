"""
Basic analysis example with synthetic data
"""

import numpy as np
import bioscience_engine as bio
from bioscience_engine.utils import create_synthetic_dataset
from PIL import Image
from pathlib import Path


def main():
    """Run basic analysis workflow"""
    
    print("="*60)
    print("BioScience Engine - Basic Analysis Example")
    print("="*60)
    
    # Create synthetic data
    print("\n1  Creating synthetic dataset...")
    create_synthetic_dataset(n_frames=30, n_cells=20)
    
    # Initialize pipeline
    print("\n2  Initializing pipeline...")
    pipeline = bio.Pipeline()
    
    # Load images
    print("\n3  Loading images...")
    pipeline.load_images("synthetic_data/")
    
    # Configure parameters
    print("\n4  Setting parameters...")
    pipeline.set_parameters(
        min_cell_size=80,
        max_cell_size=600,
        detection_threshold=0.6,
        detection_method='hybrid',
        tracking_max_distance=60.0,
        denoise_kernel=3,
        enhance_contrast=True
    )
    
    # Denoise
    print("\n  Preprocessing images...")
    pipeline.denoise()
    
    # Detect cells
    print("\n  Detecting cells...")
    pipeline.detect_cells()
    
    # Track cells
    print("\n  Tracking cells...")
    trajectories = pipeline.track_cells()
    
    # Export results
    print("\n  Exporting results...")
    pipeline.export_summary("results/")
    
    # Visualize
    print("\n  Creating visualizations...")
    pipeline.visualize(frame_idx=0, show_detections=True, show_tracks=False,
                      save_path="results/frame_000_detections.png")
    pipeline.visualize(frame_idx=15, show_detections=True, show_tracks=True,
                      save_path="results/frame_015_tracks.png")
    
    # Create video
    print("\n  Creating video...")
    pipeline.create_video("results/analysis_video.mp4", fps=5, 
                         show_detections=True, show_tracks=True)
    
    # Print summary
    print("\n" + "="*60)
    print(" Analysis Complete!")
    print("="*60)
    print(f" Tracked {len(trajectories)} cells across {trajectories.time_points} frames")
    print(f" Results saved in results/ directory")
    print(f"   - trajectories.csv")
    print(f"   - track_statistics.csv")
    print(f"   - detection_counts.csv")
    print(f"   - analysis_video.mp4")
    print("="*60)


if __name__ == "__main__":
    main()
