"""
Main analysis pipeline
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import warnings

from .detector import CellDetector
from .tracker import CellTracker, Track
from .utils import (load_image, load_tiff_stack, apply_clahe)


class Trajectory:
    """Container for cell trajectory data"""
    
    def __init__(self, tracks: List[Track]):
        self.tracks = tracks
        self.n_cells = len(tracks)
        self.time_points = max([len(t.cells) for t in tracks]) if tracks else 0
        
        self._build_array()
    
    def _build_array(self):
        if self.n_cells == 0:
            self.data = np.array([])
            return
        
        max_frames = max([max(t.frame_indices) for t in self.tracks]) + 1
        
        self.data = np.full((self.n_cells, max_frames, 4), np.nan)
        
        for i, track in enumerate(self.tracks):
            for j, (cell, frame_idx) in enumerate(zip(track.cells, track.frame_indices)):
                self.data[i, frame_idx] = cell.to_array()
    
    def __len__(self) -> int:
        return self.n_cells
    
    def to_csv(self, filename: str):
        records = []
        
        for track_idx, track in enumerate(self.tracks):
            for cell, frame_idx in zip(track.cells, track.frame_indices):
                records.append({
                    'track_id': track.track_id,
                    'cell_index': track_idx,
                    'frame': frame_idx,
                    'x': cell.x,
                    'y': cell.y,
                    'width': cell.width,
                    'height': cell.height,
                    'area': cell.area,
                    'confidence': cell.confidence,
                    'mean_intensity': cell.mean_intensity
                })
        
        df = pd.DataFrame(records)
        df.to_csv(filename, index=False)
        print(f"Saved trajectories to {filename}")
    
    def get_track(self, track_idx: int) -> Track:
        if track_idx >= self.n_cells:
            raise IndexError(f"Track {track_idx} out of range")
        return self.tracks[track_idx]
    
    def get_statistics(self) -> pd.DataFrame:
        stats = []
        
        for track in self.tracks:
            trajectory = track.get_trajectory()
            
            positions = trajectory[:, :2]
            if len(positions) > 1:
                velocities = np.diff(positions, axis=0)
                speeds = np.sqrt(np.sum(velocities**2, axis=1))
                
                avg_speed = np.mean(speeds)
                max_speed = np.max(speeds)
                path_length = np.sum(speeds)
                
                displacement = np.sqrt(np.sum((positions[-1] - positions[0])**2))
                directionality = displacement / path_length if path_length > 0 else 0
            else:
                avg_speed = max_speed = path_length = displacement = directionality = 0
            
            stats.append({
                'track_id': track.track_id,
                'n_detections': len(track.cells),
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'path_length': path_length,
                'displacement': displacement,
                'directionality': directionality
            })
        
        return pd.DataFrame(stats)


class Pipeline:
    """Main microscopy analysis pipeline"""
    
    def __init__(self):
        self.images = []
        self.processed_images = []
        self.detections = []
        self.tracker = None
        self.trajectory = None
        
        self.params = {
            'min_cell_size': 50,
            'max_cell_size': 500,
            'detection_threshold': 0.7,
            'detection_method': 'hybrid',
            'tracking_max_distance': 50.0,
            'tracking_max_age': 5,
            'tracking_min_hits': 3,
            'denoise_kernel': 5,
            'enhance_contrast': True,
            'clahe_clip_limit': 2.0
        }
        
    
    def load_images(self, path: str):
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        if path_obj.is_file():
            if path_obj.suffix.lower() in ['.tif', '.tiff']:
                self.images = load_tiff_stack(str(path_obj))
            else:
                self.images = [load_image(str(path_obj))]
        
        elif path_obj.is_dir():
            image_files = sorted(
                list(path_obj.glob("*.tif")) +
                list(path_obj.glob("*.tiff")) +
                list(path_obj.glob("*.png")) +
                list(path_obj.glob("*.jpg"))
            )
            
            if not image_files:
                raise ValueError(f"No images found in {path}")
            
            self.images = [load_image(str(f)) for f in image_files]
        
        print(f"Loaded {len(self.images)} images")
        if self.images:
            h, w = self.images[0].shape
            print(f"Image size: {w}x{h} pixels")
    
    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                print(f"{key} = {value}")
            else:
                warnings.warn(f"Unknown parameter: {key}")
    
    def denoise(self):
        if not self.images:
            raise RuntimeError("No images loaded")
        
        print("Denoising images...")
        self.processed_images = []
        
        kernel_size = self.params['denoise_kernel']
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        for i, img in enumerate(self.images):
            denoised = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
            if self.params['enhance_contrast']:
                denoised = apply_clahe(denoised, 
                                      clip_limit=self.params['clahe_clip_limit'])
            
            self.processed_images.append(denoised)
            
            if (i + 1) % 10 == 0 or i == len(self.images) - 1:
                print(f"Processed {i + 1}/{len(self.images)} images")
        
        print(f"Denoised {len(self.processed_images)} images")
    
    def detect_cells(self):
        images_to_process = self.processed_images if self.processed_images else self.images
        
        if not images_to_process:
            raise RuntimeError("No images to process")
        
        print(f"Detecting cells using '{self.params['detection_method']}' method")
        
        detector = CellDetector(
            min_size=self.params['min_cell_size'],
            max_size=self.params['max_cell_size'],
            threshold=self.params['detection_threshold']
        )
        
        self.detections = []
        total_cells = 0
        
        for i, img in enumerate(images_to_process):
            cells = detector.detect(img, method=self.params['detection_method'])
            self.detections.append(cells)
            total_cells += len(cells)
            
            if (i + 1) % 10 == 0 or i == len(images_to_process) - 1:
                print(f"Frame {i + 1}/{len(images_to_process)}: {len(cells)} cells detected")
        
        avg_cells = total_cells / len(self.detections) if self.detections else 0
        print(f"Detected {total_cells} total cells (avg: {avg_cells:.1f} per frame)")
    
    def track_cells(self) -> Trajectory:
        if not self.detections:
            raise RuntimeError("No cells detected. Run detect_cells() first")
        
        print("Tracking cells across frames")
        
        self.tracker = CellTracker(
            max_distance=self.params['tracking_max_distance'],
            max_age=self.params['tracking_max_age'],
            min_hits=self.params['tracking_min_hits']
        )
        
        for i, cells in enumerate(self.detections):
            tracks = self.tracker.update(cells)
            
            if (i + 1) % 10 == 0 or i == len(self.detections) - 1:
                print(f"Frame {i + 1}/{len(self.detections)}: {len(tracks)} active tracks")
        
        confirmed_tracks = [t for t in self.tracker.tracks 
                           if t.is_confirmed(self.params['tracking_min_hits'])]
        
        self.trajectory = Trajectory(confirmed_tracks)
        
        print(f"Tracked {len(self.trajectory)} unique cells")
        print(f"Average detections per track: {np.mean([len(t.cells) for t in confirmed_tracks]):.1f}")
        
        return self.trajectory
    
    def visualize(self, frame_idx: int = 0, show_detections: bool = True, 
                  show_tracks: bool = False, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle
        
        if frame_idx >= len(self.images):
            raise IndexError(f"Frame {frame_idx} out of range")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        if self.processed_images:
            ax.imshow(self.processed_images[frame_idx], cmap='gray')
        else:
            ax.imshow(self.images[frame_idx], cmap='gray')
        
        if show_detections and frame_idx < len(self.detections):
            for cell in self.detections[frame_idx]:
                x1, y1, x2, y2 = cell.get_bbox()
                rect = Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='lime', facecolor='none',
                               alpha=0.7)
                ax.add_patch(rect)
                
                ax.plot(cell.x, cell.y, 'r+', markersize=10, markeredgewidth=2)
        
        if show_tracks and self.trajectory:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(self.trajectory.tracks)))
            
            for track, color in zip(self.trajectory.tracks, colors):
                if frame_idx in track.frame_indices:
                    positions = []
                    for cell, fidx in zip(track.cells, track.frame_indices):
                        if fidx <= frame_idx:
                            positions.append([cell.x, cell.y])
                    
                    if len(positions) > 1:
                        positions = np.array(positions)
                        ax.plot(positions[:, 0], positions[:, 1], 
                               color=color, linewidth=2, alpha=0.7)
                    
                    idx = track.frame_indices.index(frame_idx)
                    cell = track.cells[idx]
                    ax.plot(cell.x, cell.y, 'o', color=color, markersize=8)
                    ax.text(cell.x + 10, cell.y, f'#{track.track_id}',
                           color=color, fontsize=9, weight='bold')
        
        ax.set_title(f"Frame {frame_idx} - {len(self.detections[frame_idx]) if frame_idx < len(self.detections) else 0} cells detected",
                    fontsize=14, weight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def create_video(self, output_path: str, fps: int = 10, 
                    show_detections: bool = True, show_tracks: bool = True):
        if not self.images:
            raise RuntimeError("No images loaded")
        
        print(f"Creating video: {output_path}")
        
        h, w = self.images[0].shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        colors_bgr = [
            tuple(map(int, np.array(plt.cm.rainbow(i / len(self.trajectory.tracks))[:3]) * 255))
            if self.trajectory else (0, 255, 0)
            for i in range(len(self.trajectory.tracks) if self.trajectory else 1)
        ]
        
        for frame_idx in range(len(self.images)):
            if self.processed_images:
                img = self.processed_images[frame_idx]
            else:
                img = self.images[frame_idx]
            
            frame = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            if show_detections and frame_idx < len(self.detections):
                for cell in self.detections[frame_idx]:
                    x1, y1, x2, y2 = cell.get_bbox()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (int(cell.x), int(cell.y)), 3, (0, 0, 255), -1)
            
            if show_tracks and self.trajectory:
                for track_idx, track in enumerate(self.trajectory.tracks):
                    color = colors_bgr[track_idx % len(colors_bgr)]
                    
                    if frame_idx in track.frame_indices:
                        positions = []
                        for cell, fidx in zip(track.cells, track.frame_indices):
                            if fidx <= frame_idx:
                                positions.append((int(cell.x), int(cell.y)))
                        
                        if len(positions) > 1:
                            for i in range(len(positions) - 1):
                                cv2.line(frame, positions[i], positions[i+1], color, 2)
                        
                        if positions:
                            cv2.circle(frame, positions[-1], 5, color, -1)
                            cv2.putText(frame, f'#{track.track_id}',
                                      (positions[-1][0] + 10, positions[-1][1]),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.putText(frame, f'Frame {frame_idx}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
            
            if (frame_idx + 1) % 10 == 0 or frame_idx == len(self.images) - 1:
                print(f"Rendered {frame_idx + 1}/{len(self.images)} frames")
        
        out.release()
        print(f"Video saved to {output_path}")
    
    def export_summary(self, output_dir: str = "."):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Exporting analysis summary to {output_dir}/")
        
        if self.trajectory:
            self.trajectory.to_csv(output_path / "trajectories.csv")
            
            stats_df = self.trajectory.get_statistics()
            stats_df.to_csv(output_path / "track_statistics.csv", index=False)
            print(f"Saved track statistics")
        
        if self.detections:
            detection_counts = pd.DataFrame({
                'frame': range(len(self.detections)),
                'n_cells': [len(cells) for cells in self.detections]
            })
            detection_counts.to_csv(output_path / "detection_counts.csv", index=False)
            print(f"Saved detection counts")
        
        params_df = pd.DataFrame([self.params]).T
        params_df.columns = ['value']
        params_df.to_csv(output_path / "parameters.csv")
        print(f"Saved analysis parameters")
        
        print("Export complete")
        