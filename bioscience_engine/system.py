"""
Complete Cell Tracking Analysis System
Integrates all modules into a comprehensive analysis platform

This is the master script that ties together:
- Cell detection (fixed)
- Cell tracking
- Division detection
- Apoptosis detection
- Multi-channel analysis
- Morphology analysis
- Statistical analysis
- Interactive GUI
"""

from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd


class System:
    """
    Master class integrating all analysis modules
    
    Complete workflow:
    1. Load images (single/multi-channel, 2D/3D)
    2. Preprocess (denoise, enhance)
    3. Detect cells (blob/watershed/contour/hybrid)
    4. Track cells over time
    5. Detect divisions
    6. Detect apoptosis
    7. Analyze morphology
    8. Measure fluorescence (multi-channel)
    9. Statistical analysis
    10. Export results
    """
    
    def __init__(self, output_dir: str = "./results"):
        """
        Initialize complete system
        
        Args:
            output_dir: Directory for all outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Core modules
        from bioscience_engine.pipeline import Pipeline
        self.pipeline = Pipeline()
        
        # Analysis modules
        self.division_detector = None
        self.apoptosis_detector = None
        self.morphology_analyzer = None
        self.multichannel_analyzer = None
        self.stats_analyzer = None
        
        # Results storage
        self.results = {
            'trajectory': None,
            'divisions': [],
            'apoptosis': [],
            'morphology': {},
            'fluorescence': {},
            'statistics': {}
        }
        
        print("=" * 70)
        print("COMPLETE CELL TRACKING ANALYSIS SYSTEM")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        print()
    
    def load_images(self, path: str):
        """Load microscopy images"""
        print("📁 LOADING IMAGES")
        print("-" * 70)
        self.pipeline.load_images(path)
        print()
    
    def set_parameters(self, **kwargs):
        """Set analysis parameters"""
        print("⚙️  CONFIGURING PARAMETERS")
        print("-" * 70)
        self.pipeline.set_parameters(**kwargs)
        print()
    
    def run_complete_analysis(self,
                            detect_divisions: bool = True,
                            detect_apoptosis: bool = True,
                            analyze_morphology: bool = True,
                            run_statistics: bool = True):
        """
        Run complete analysis pipeline
        
        Args:
            detect_divisions: Enable division detection
            detect_apoptosis: Enable apoptosis detection
            analyze_morphology: Enable morphology analysis
            run_statistics: Enable statistical analysis
        """
        print("\n" + "=" * 70)
        print("RUNNING COMPLETE ANALYSIS PIPELINE")
        print("=" * 70)
        
        # Step 1: Preprocessing
        print("\n1️⃣  PREPROCESSING")
        print("-" * 70)
        self.pipeline.denoise()
        
        # Step 2: Detection
        print("\n2️⃣  CELL DETECTION")
        print("-" * 70)
        self.pipeline.detect_cells()
        
        # Step 3: Tracking
        print("\n3️⃣  CELL TRACKING")
        print("-" * 70)
        trajectory = self.pipeline.track_cells()
        self.results['trajectory'] = trajectory
        
        # Step 4: Division detection
        if detect_divisions:
            print("\n4️⃣  DIVISION DETECTION")
            print("-" * 70)
            self._detect_divisions()
        
        # Step 5: Apoptosis detection
        if detect_apoptosis:
            print("\n5️⃣  APOPTOSIS DETECTION")
            print("-" * 70)
            self._detect_apoptosis()
        
        # Step 6: Morphology analysis
        if analyze_morphology:
            print("\n6️⃣  MORPHOLOGY ANALYSIS")
            print("-" * 70)
            self._analyze_morphology()
        
        # Step 7: Statistics
        if run_statistics:
            print("\n7️⃣  STATISTICAL ANALYSIS")
            print("-" * 70)
            self._run_statistics()
        
        print("\n✅ COMPLETE ANALYSIS FINISHED!")
        print("=" * 70)
    
    def _detect_divisions(self):
        """Internal: Run division detection"""
        try:
            from division_detector import CellDivisionDetector
            
            if self.division_detector is None:
                self.division_detector = CellDivisionDetector()
            
            divisions = self.division_detector.detect_divisions(
                self.results['trajectory'].tracks,
                self.pipeline.detections
            )
            
            self.results['divisions'] = divisions
            print(f"  Detected {len(divisions)} cell division events")
            
            # Visualize first division
            if divisions:
                vis = self.division_detector.visualize_divisions(
                    self.pipeline.images[divisions[0].division_frame],
                    [divisions[0]],
                    self.results['trajectory'].tracks
                )
                import cv2
                cv2.imwrite(str(self.output_dir / "division_example.png"), vis)
            
        except Exception as e:
            print(f"  ⚠️  Division detection error: {e}")
    
    def _detect_apoptosis(self):
        """Internal: Run apoptosis detection"""
        try:
            from apoptosis_detector import ApoptosisDetector
            
            if self.apoptosis_detector is None:
                self.apoptosis_detector = ApoptosisDetector()
            
            apoptosis_events = self.apoptosis_detector.detect_apoptosis(
                self.results['trajectory'].tracks,
                self.pipeline.images
            )
            
            self.results['apoptosis'] = apoptosis_events
            print(f"  Detected {len(apoptosis_events)} apoptosis events")
            
            # Calculate survival
            survival_stats = self.apoptosis_detector.calculate_survival_statistics(
                self.results['trajectory'].tracks,
                apoptosis_events
            )
            
            print(f"  Survival rate: {survival_stats['survival_rate']:.1%}")
            
        except Exception as e:
            print(f"  ⚠️  Apoptosis detection error: {e}")
    
    def _analyze_morphology(self):
        """Internal: Run morphology analysis"""
        try:
            from morphology_analyzer import MorphologyAnalyzer
            
            if self.morphology_analyzer is None:
                self.morphology_analyzer = MorphologyAnalyzer()
            
            # Analyze first frame
            if self.pipeline.detections:
                pop_stats = self.morphology_analyzer.analyze_population(
                    self.pipeline.detections[0],
                    self.pipeline.images[0]
                )
                
                self.results['morphology'] = pop_stats
                
                print(f"  Analyzed {len(self.pipeline.detections[0])} cells")
                print(f"  Mean circularity: {pop_stats['circularity']['mean']:.3f}")
                print(f"  Morphology distribution:")
                for mtype, count in pop_stats['morphology_distribution'].items():
                    print(f"    {mtype}: {count} cells")
                
                # Create visualization
                vis = self.morphology_analyzer.visualize_morphology(
                    self.pipeline.detections[0],
                    self.pipeline.images[0],
                    feature='circularity'
                )
                import cv2
                cv2.imwrite(str(self.output_dir / "morphology_circularity.png"), vis)
                
        except Exception as e:
            print(f"  ⚠️  Morphology analysis error: {e}")
    
    def _run_statistics(self):
        """Internal: Run statistical analysis"""
        try:
            from statistical_analyzer import StatisticalAnalyzer
            
            if self.stats_analyzer is None:
                self.stats_analyzer = StatisticalAnalyzer()
            
            # Calculate track statistics
            track_stats = self.results['trajectory'].get_statistics()
            
            print(f"  Tracks analyzed: {len(track_stats)}")
            print(f"  Mean speed: {track_stats['avg_speed'].mean():.2f} ± {track_stats['avg_speed'].std():.2f}")
            print(f"  Mean path length: {track_stats['path_length'].mean():.2f}")
            print(f"  Mean displacement: {track_stats['displacement'].mean():.2f}")
            
            self.results['statistics'] = track_stats
            
        except Exception as e:
            print(f"  ⚠️  Statistical analysis error: {e}")
    
    def load_multichannel(self, 
                         channels: Dict[str, str],
                         auto_threshold: bool = True):
        """
        Load multi-channel fluorescence data
        
        Args:
            channels: Dictionary mapping channel name to file path
            auto_threshold: Automatically calculate thresholds
            
        Example:
            system.load_multichannel({
                'gfp': 'path/to/gfp_images/',
                'rfp': 'path/to/rfp_images/',
                'dapi': 'path/to/dapi_images/'
            })
        """
        print("\n🔬 LOADING MULTI-CHANNEL DATA")
        print("-" * 70)
        
        try:
            from multichannel_analyzer import MultiChannelAnalyzer
            
            if self.multichannel_analyzer is None:
                self.multichannel_analyzer = MultiChannelAnalyzer()
            
            # Load each channel
            for name, path in channels.items():
                # Load images from path
                import cv2
                images = []
                path_obj = Path(path)
                
                if path_obj.is_file():
                    img = cv2.imread(str(path_obj), cv2.IMREAD_GRAYSCALE)
                    images = [img.astype(np.float32) / 255.0]
                else:
                    # Load directory
                    image_files = sorted(list(path_obj.glob("*.tif")) +
                                       list(path_obj.glob("*.tiff")) +
                                       list(path_obj.glob("*.png")))
                    images = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 
                             for f in image_files]
                
                threshold = None if auto_threshold else 0.5
                self.multichannel_analyzer.load_channel(name, images, threshold)
            
            print(f"  Loaded {len(channels)} channels")
            print()
            
        except Exception as e:
            print(f"  ⚠️  Multi-channel loading error: {e}")
    
    def analyze_fluorescence(self):
        """Analyze fluorescence in all channels"""
        if self.multichannel_analyzer is None:
            print("⚠️  No multi-channel data loaded")
            return
        
        print("\n🔬 FLUORESCENCE ANALYSIS")
        print("-" * 70)
        
        # Measure fluorescence for all tracks
        self.multichannel_analyzer.export_fluorescence_data(
            self.results['trajectory'].tracks,
            str(self.output_dir / "fluorescence_measurements.csv")
        )
        
        # Calculate colocalization
        channels = list(self.multichannel_analyzer.channels.keys())
        if len(channels) >= 2:
            coloc = self.multichannel_analyzer.calculate_colocalization(
                channels[0], channels[1],
                cells=self.pipeline.detections[0] if self.pipeline.detections else None
            )
            
            print(f"  Colocalization ({channels[0]} / {channels[1]}):")
            print(f"    Pearson: {coloc.pearson_correlation:.3f}")
            print(f"    Manders M1: {coloc.manders_m1:.3f}")
            print(f"    Manders M2: {coloc.manders_m2:.3f}")
        
        print()
    
    def export_all_results(self):
        """Export all analysis results"""
        print("\n💾 EXPORTING RESULTS")
        print("-" * 70)
        
        # Trajectories
        if self.results['trajectory']:
            self.results['trajectory'].to_csv(
                str(self.output_dir / "trajectories.csv")
            )
            print("  ✓ Trajectories exported")
        
        # Track statistics
        if self.results['statistics'] is not None:
            if isinstance(self.results['statistics'], pd.DataFrame):
                self.results['statistics'].to_csv(
                    str(self.output_dir / "track_statistics.csv"),
                    index=False
                )
                print("  ✓ Track statistics exported")
        
        # Divisions
        if self.results['divisions']:
            division_data = [{
                'parent_track': d.parent_track_id,
                'daughter1_track': d.daughter_track_ids[0],
                'daughter2_track': d.daughter_track_ids[1],
                'frame': d.division_frame,
                'confidence': d.confidence
            } for d in self.results['divisions']]
            
            pd.DataFrame(division_data).to_csv(
                str(self.output_dir / "divisions.csv"),
                index=False
            )
            print("  ✓ Division events exported")
        
        # Apoptosis
        if self.results['apoptosis']:
            apoptosis_data = [{
                'track_id': e.track_id,
                'death_frame': e.death_frame,
                'death_type': e.death_type,
                'confidence': e.confidence
            } for e in self.results['apoptosis']]
            
            pd.DataFrame(apoptosis_data).to_csv(
                str(self.output_dir / "apoptosis.csv"),
                index=False
            )
            print("  ✓ Apoptosis events exported")
        
        # Morphology
        if self.results['morphology']:
            if self.morphology_analyzer:
                self.morphology_analyzer.export_features(
                    self.pipeline.detections[0],
                    self.pipeline.images[0],
                    str(self.output_dir / "morphology_features.csv")
                )
                print("  ✓ Morphology features exported")
        
        # Complete summary
        self.pipeline.export_summary(str(self.output_dir))
        
        print(f"\n  All results saved to: {self.output_dir}")
        print()
    
    def create_visualizations(self):
        """Create all visualization outputs"""
        print("\n📊 CREATING VISUALIZATIONS")
        print("-" * 70)
        
        # Video
        try:
            video_path = str(self.output_dir / "analysis_video.mp4")
            self.pipeline.create_video(
                video_path,
                fps=10,
                show_detections=True,
                show_tracks=True
            )
            print("  ✓ Analysis video created")
        except Exception as e:
            print(f"  ⚠️  Video creation error: {e}")
        
        # Static visualizations
        for i in [0, len(self.pipeline.images) // 2, len(self.pipeline.images) - 1]:
            try:
                img_path = str(self.output_dir / f"frame_{i:04d}.png")
                self.pipeline.visualize(
                    i,
                    show_detections=True,
                    show_tracks=True,
                    save_path=img_path
                )
            except:
                pass
        
        print("  ✓ Static images created")
        print()
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n📄 GENERATING FINAL REPORT")
        print("-" * 70)
        
        report_path = self.output_dir / "ANALYSIS_REPORT.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("COMPREHENSIVE CELL TRACKING ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Dataset info
            f.write("DATASET INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Number of frames: {len(self.pipeline.images)}\n")
            f.write(f"Image size: {self.pipeline.images[0].shape}\n\n")
            
            # Tracking results
            if self.results['trajectory']:
                f.write("TRACKING RESULTS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total unique cells tracked: {len(self.results['trajectory'])}\n")
                
                stats = self.results['trajectory'].get_statistics()
                f.write(f"Average track length: {stats['n_detections'].mean():.1f} frames\n")
                f.write(f"Average speed: {stats['avg_speed'].mean():.2f}\n")
                f.write(f"Average displacement: {stats['displacement'].mean():.2f}\n\n")
            
            # Division events
            if self.results['divisions']:
                f.write("CELL DIVISION EVENTS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total divisions detected: {len(self.results['divisions'])}\n")
                f.write(f"Average division confidence: {np.mean([d.confidence for d in self.results['divisions']]):.2f}\n\n")
            
            # Apoptosis events
            if self.results['apoptosis']:
                f.write("CELL DEATH EVENTS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total apoptosis events: {len(self.results['apoptosis'])}\n")
                
                death_types = {}
                for e in self.results['apoptosis']:
                    death_types[e.death_type] = death_types.get(e.death_type, 0) + 1
                
                for dtype, count in death_types.items():
                    f.write(f"  {dtype}: {count}\n")
                f.write("\n")
            
            # Morphology
            if self.results['morphology']:
                f.write("MORPHOLOGY ANALYSIS\n")
                f.write("-" * 70 + "\n")
                morph_dist = self.results['morphology'].get('morphology_distribution', {})
                for mtype, count in morph_dist.items():
                    f.write(f"  {mtype}: {count} cells\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")
        
        print(f"  ✓ Report saved to {report_path}")
        print()


def example_complete_workflow():
    """Example of complete analysis workflow"""
    
    # Initialize system
    system = System(output_dir="./complete_analysis_results")
    
    # Load images
    system.load_images("path/to/your/images")
    
    # Configure parameters
    system.set_parameters(
        min_cell_size=100,
        max_cell_size=3000,
        detection_threshold=0.6,
        detection_method='hybrid',
        tracking_max_distance=50.0,
        enhance_contrast=True
    )
    
    # Run complete analysis
    system.run_complete_analysis(
        detect_divisions=True,
        detect_apoptosis=True,
        analyze_morphology=True,
        run_statistics=True
    )
    
    # Optional: Multi-channel analysis
    # system.load_multichannel({
    #     'gfp': 'path/to/gfp',
    #     'rfp': 'path/to/rfp'
    # })
    # system.analyze_fluorescence()
    
    # Export everything
    system.export_all_results()
    
    # Create visualizations
    system.create_visualizations()
    
    # Generate report
    system.generate_final_report()
    
    print("\n🎉 COMPLETE ANALYSIS FINISHED!")
    print(f"Check results in: {system.output_dir}")


if __name__ == "__main__":
    example_complete_workflow()