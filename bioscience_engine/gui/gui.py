import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext
import threading
from pathlib import Path
import sys
from PIL import Image, ImageTk
import numpy as np
import cv2


class CellTrackingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BioScience Engine Studio - Professional Cell Analysis Tool")
        self.root.geometry("1400x900")
        
        self.images_path = None
        self.pipeline = None
        self.current_frame = 0
        self.analysis_running = False
        
        self.params = {
            'min_cell_size': tk.IntVar(value=100),
            'max_cell_size': tk.IntVar(value=3000),
            'detection_threshold': tk.DoubleVar(value=0.6),
            'detection_method': tk.StringVar(value='hybrid'),
            'tracking_max_distance': tk.DoubleVar(value=50.0),
            'tracking_max_age': tk.IntVar(value=5),
            'tracking_min_hits': tk.IntVar(value=3),
            'denoise_kernel': tk.IntVar(value=5),
            'enhance_contrast': tk.BooleanVar(value=True),
            'clahe_clip_limit': tk.DoubleVar(value=2.0)
        }
        
        self._create_widgets()
        
    def _create_widgets(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_panel = ttk.Frame(main_paned, width=350)
        main_paned.add(left_panel, weight=0)
        
        right_panel = ttk.Frame(main_paned)
        main_paned.add(right_panel, weight=1)
        
        self._create_left_panel(left_panel)
        self._create_right_panel(right_panel)
        
    def _create_left_panel(self, parent):
        title = ttk.Label(parent, text="BioScience Engine Studio", 
                         font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5)
        
        file_tab = ttk.Frame(notebook)
        notebook.add(file_tab, text="Load Data")
        self._create_file_tab(file_tab)
        
        detect_tab = ttk.Frame(notebook)
        notebook.add(detect_tab, text="Detection")
        self._create_detection_tab(detect_tab)
        
        track_tab = ttk.Frame(notebook)
        notebook.add(track_tab, text="Tracking")
        self._create_tracking_tab(track_tab)
        
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text="Analysis")
        self._create_analysis_tab(analysis_tab)
        
        console_frame = ttk.LabelFrame(parent, text="Console Output")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.console = scrolledtext.ScrolledText(console_frame, height=8, 
                                                 state='disabled',
                                                 bg='black', fg='lime',
                                                 font=('Courier', 9))
        self.console.pack(fill=tk.BOTH, expand=True)
        
        
    def _create_file_tab(self, parent):
        load_frame = ttk.Frame(parent)
        load_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(load_frame, text="Load Images",
                  command=self.load_images,
                  style='Accent.TButton').pack(fill=tk.X)
        
        self.path_label = ttk.Label(parent, text="No images loaded",
                                    wraplength=300, foreground='gray')
        self.path_label.pack(pady=5)
        
        info_frame = ttk.LabelFrame(parent, text="Dataset Info")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.info_label = ttk.Label(info_frame, text="", justify=tk.LEFT)
        self.info_label.pack(padx=5, pady=5)
        
    def _create_detection_tab(self, parent):
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self._add_parameter_slider(scrollable_frame, "Min Cell Size (px²)",
                                   self.params['min_cell_size'],
                                   from_=10, to=1000)
        
        self._add_parameter_slider(scrollable_frame, "Max Cell Size (px²)",
                                   self.params['max_cell_size'],
                                   from_=100, to=10000)
        
        self._add_parameter_slider(scrollable_frame, "Detection Threshold",
                                   self.params['detection_threshold'],
                                   from_=0.1, to=1.0, resolution=0.05)
        
        method_frame = ttk.LabelFrame(scrollable_frame, text="Detection Method")
        method_frame.pack(fill=tk.X, padx=10, pady=5)
        
        for method in ['blob', 'watershed', 'contour', 'hybrid']:
            ttk.Radiobutton(method_frame, text=method.capitalize(),
                           variable=self.params['detection_method'],
                           value=method).pack(anchor=tk.W, padx=10)
        
        preproc_frame = ttk.LabelFrame(scrollable_frame, text="Preprocessing")
        preproc_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Checkbutton(preproc_frame, text="Enhance Contrast (CLAHE)",
                       variable=self.params['enhance_contrast']).pack(anchor=tk.W, padx=10)
        
        self._add_parameter_slider(preproc_frame, "CLAHE Clip Limit",
                                   self.params['clahe_clip_limit'],
                                   from_=1.0, to=5.0, resolution=0.5)
        
        self._add_parameter_slider(preproc_frame, "Denoise Kernel Size",
                                   self.params['denoise_kernel'],
                                   from_=3, to=15, resolution=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def _create_tracking_tab(self, parent):
        self._add_parameter_slider(parent, "Max Distance (pixels)",
                                   self.params['tracking_max_distance'],
                                   from_=10, to=200, resolution=5)
        
        self._add_parameter_slider(parent, "Max Age (frames)",
                                   self.params['tracking_max_age'],
                                   from_=1, to=20)
        
        self._add_parameter_slider(parent, "Min Hits (confirmations)",
                                   self.params['tracking_min_hits'],
                                   from_=1, to=10)
        
        ttk.Button(parent, text="Preview Detection",
                  command=self.preview_detection).pack(pady=10, padx=10, fill=tk.X)
        
    def _create_analysis_tab(self, parent):
        self.run_button = ttk.Button(parent, text="RUN FULL ANALYSIS",
                                     command=self.run_analysis,
                                     style='Accent.TButton')
        self.run_button.pack(pady=10, padx=10, fill=tk.X)
        
        self.progress = ttk.Progressbar(parent, mode='indeterminate')
        self.progress.pack(pady=5, padx=10, fill=tk.X)
        
        export_frame = ttk.LabelFrame(parent, text="Export Results")
        export_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(export_frame, text="Export CSV",
                  command=self.export_csv).pack(pady=5, fill=tk.X, padx=5)
        
        ttk.Button(export_frame, text="Create Video",
                  command=self.create_video).pack(pady=5, fill=tk.X, padx=5)
        
        ttk.Button(export_frame, text="Generate Report",
                  command=self.generate_report).pack(pady=5, fill=tk.X, padx=5)
        
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Analysis")
        advanced_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(advanced_frame, text="Detect Cell Divisions",
                  command=self.detect_divisions).pack(pady=5, fill=tk.X, padx=5)
        
        ttk.Button(advanced_frame, text="Detect Apoptosis",
                  command=self.detect_apoptosis).pack(pady=5, fill=tk.X, padx=5)
        
        ttk.Button(advanced_frame, text="Morphology Analysis",
                  command=self.analyze_morphology).pack(pady=5, fill=tk.X, padx=5)
        
    def _create_right_panel(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Frame:").pack(side=tk.LEFT, padx=5)
        
        self.frame_slider = ttk.Scale(control_frame, from_=0, to=0,
                                     orient=tk.HORIZONTAL,
                                     command=self.update_frame)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.frame_label = ttk.Label(control_frame, text="0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        
        vis_frame = ttk.Frame(parent)
        vis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.show_detections = tk.BooleanVar(value=True)
        self.show_tracks = tk.BooleanVar(value=False)
        self.show_ids = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(vis_frame, text="Show Detections",
                       variable=self.show_detections,
                       command=self.update_visualization).pack(side=tk.LEFT, padx=5)
        
        ttk.Checkbutton(vis_frame, text="Show Tracks",
                       variable=self.show_tracks,
                       command=self.update_visualization).pack(side=tk.LEFT, padx=5)
        
        ttk.Checkbutton(vis_frame, text="Show IDs",
                       variable=self.show_ids,
                       command=self.update_visualization).pack(side=tk.LEFT, padx=5)
        
        canvas_frame = ttk.Frame(parent, relief=tk.SUNKEN)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        stats_frame = ttk.LabelFrame(parent, text="Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.stats_label = ttk.Label(stats_frame, text="No analysis yet",
                                     justify=tk.LEFT)
        self.stats_label.pack(padx=10, pady=5)
        
    def _add_parameter_slider(self, parent, label, variable, 
                             from_, to, resolution=1):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(frame, text=label).pack(anchor=tk.W)
        
        slider_frame = ttk.Frame(frame)
        slider_frame.pack(fill=tk.X)
        
        slider = ttk.Scale(slider_frame, from_=from_, to=to,
                          variable=variable, orient=tk.HORIZONTAL)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        value_label = ttk.Label(slider_frame, text=f"{variable.get():.2f}",
                               width=8)
        value_label.pack(side=tk.LEFT, padx=5)
        
        def update_label(*args):
            value_label.config(text=f"{variable.get():.2f}")
        
        variable.trace('w', update_label)
        
    def load_images(self):
        path = filedialog.askdirectory(title="Select Image Directory")
        
        if not path:
            return
        
        self.log(f"Loading images from {path}...")
        
        try:
            from bioscience_engine.pipeline import Pipeline
            
            self.pipeline = Pipeline()
            self.pipeline.load_images(path)
            
            self.images_path = path
            self.path_label.config(text=path, foreground='black')
            
            n_images = len(self.pipeline.images)
            h, w = self.pipeline.images[0].shape
            
            info = f"Images: {n_images}\nSize: {w}x{h} pixels"
            self.info_label.config(text=info)
            
            self.frame_slider.config(to=n_images-1)
            self.frame_label.config(text=f"0/{n_images-1}")
            
            self.log(f"Loaded {n_images} images successfully!")
            
            self.display_image(0)
            
        except Exception as e:
            self.log(f"Error loading images: {e}")
            messagebox.showerror("Error", f"Failed to load images:\n{e}")
    
    def preview_detection(self):
        if self.pipeline is None or not self.pipeline.images:
            messagebox.showwarning("Warning", "Please load images first!")
            return
        
        self.log("Running detection preview...")
        
        try:
            self._apply_parameters()
            
            self.pipeline.denoise()
            self.pipeline.detect_cells()
            
            n_cells = len(self.pipeline.detections[self.current_frame])
            self.log(f"Detected {n_cells} cells in frame {self.current_frame}")
            
            self.update_visualization()
            
        except Exception as e:
            self.log(f"Detection failed: {e}")
            messagebox.showerror("Error", f"Detection failed:\n{e}")
    
    def run_analysis(self):
        if self.pipeline is None:
            messagebox.showwarning("Warning", "Please load images first!")
            return
        
        if self.analysis_running:
            messagebox.showinfo("Info", "Analysis already running!")
            return
        
        self.analysis_running = True
        self.run_button.config(state='disabled')
        self.progress.start()
        
        thread = threading.Thread(target=self._run_analysis_thread)
        thread.daemon = True
        thread.start()
    
    def _run_analysis_thread(self):
        try:
            self.log("Starting full analysis...")
            
            self._apply_parameters()
            
            self.log("Step 1/3: Preprocessing...")
            self.pipeline.denoise()
            
            self.log("Step 2/3: Detecting cells...")
            self.pipeline.detect_cells()
            
            self.log("Step 3/3: Tracking cells...")
            trajectory = self.pipeline.track_cells()
            
            self.log(f"Analysis complete!")
            self.log(f"Tracked {len(trajectory)} unique cells")
            
            self.root.after(0, self._finish_analysis)
            
        except Exception as e:
            self.log(f"Analysis failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", 
                                                            f"Analysis failed:\n{e}"))
            self.root.after(0, self._finish_analysis)
    
    def _finish_analysis(self):
        self.analysis_running = False
        self.run_button.config(state='normal')
        self.progress.stop()
        self.update_visualization()
        self.update_statistics()
    
    def _apply_parameters(self):
        params = {k: v.get() for k, v in self.params.items()}
        self.pipeline.set_parameters(**params)
    
    def update_frame(self, value):
        self.current_frame = int(float(value))
        n_frames = len(self.pipeline.images) if self.pipeline else 0
        self.frame_label.config(text=f"{self.current_frame}/{n_frames-1}")
        self.update_visualization()
    
    def update_visualization(self):
        if self.pipeline is None or not self.pipeline.images:
            return
        
        if self.pipeline.processed_images:
            img = self.pipeline.processed_images[self.current_frame]
        else:
            img = self.pipeline.images[self.current_frame]
        
        img_uint8 = (img * 255).astype(np.uint8)
        img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        
        if self.show_detections.get() and self.pipeline.detections:
            if self.current_frame < len(self.pipeline.detections):
                for cell in self.pipeline.detections[self.current_frame]:
                    x1, y1, x2, y2 = cell.get_bbox()
                    cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(img_color, (int(cell.x), int(cell.y)), 3, (0, 0, 255), -1)
                    
                    if self.show_ids.get():
                        cv2.putText(img_color, f"{id(cell) % 1000}",
                                   (int(cell.x)+10, int(cell.y)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if self.show_tracks.get() and self.pipeline.trajectory:
            import matplotlib.cm as cm
            colors = cm.rainbow(np.linspace(0, 1, len(self.pipeline.trajectory.tracks)))
            
            for track, color in zip(self.pipeline.trajectory.tracks, colors):
                if self.current_frame in track.frame_indices:
                    positions = []
                    for cell, fidx in zip(track.cells, track.frame_indices):
                        if fidx <= self.current_frame:
                            positions.append((int(cell.x), int(cell.y)))
                    
                    if len(positions) > 1:
                        color_bgr = tuple(int(c * 255) for c in reversed(color[:3]))
                        for i in range(len(positions) - 1):
                            cv2.line(img_color, positions[i], positions[i+1], color_bgr, 2)
        
        self.display_cv_image(img_color)
    
    def display_image(self, frame_idx):
        if self.pipeline is None or not self.pipeline.images:
            return
        
        img = self.pipeline.images[frame_idx]
        img_uint8 = (img * 255).astype(np.uint8)
        img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        
        self.display_cv_image(img_color)
    
    def display_cv_image(self, cv_image):
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            h, w = image_rgb.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            image_rgb = cv2.resize(image_rgb, (new_w, new_h))
        
        image_pil = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image_pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo
    
    def update_statistics(self):
        if self.pipeline is None or self.pipeline.trajectory is None:
            return
        
        stats_df = self.pipeline.trajectory.get_statistics()
        
        stats_text = f"Total Tracks: {len(self.pipeline.trajectory)}\n"
        stats_text += f"Avg Speed: {stats_df['avg_speed'].mean():.2f}\n"
        stats_text += f"Avg Path Length: {stats_df['path_length'].mean():.2f}"
        
        self.stats_label.config(text=stats_text)
    
    def export_csv(self):
        if self.pipeline is None or self.pipeline.trajectory is None:
            messagebox.showwarning("Warning", "Run analysis first!")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if filepath:
            self.pipeline.trajectory.to_csv(filepath)
            self.log(f"Exported to {filepath}")
            messagebox.showinfo("Success", "Data exported successfully!")
    
    def create_video(self):
        if self.pipeline is None:
            messagebox.showwarning("Warning", "Load images first!")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")]
        )
        
        if filepath:
            self.log("Creating video...")
            try:
                self.pipeline.create_video(filepath, fps=10,
                                          show_detections=True,
                                          show_tracks=True)
                self.log(f"Video saved to {filepath}")
                messagebox.showinfo("Success", "Video created successfully!")
            except Exception as e:
                self.log(f"Video creation failed: {e}")
                messagebox.showerror("Error", f"Failed to create video:\n{e}")
    
    def generate_report(self):
        if self.pipeline is None:
            messagebox.showwarning("Warning", "Run analysis first!")
            return
        
        directory = filedialog.askdirectory(title="Select Output Directory")
        
        if directory:
            self.log("Generating report...")
            try:
                self.pipeline.export_summary(directory)
                self.log(f"Report saved to {directory}")
                messagebox.showinfo("Success", "Report generated successfully!")
            except Exception as e:
                self.log(f"Report generation failed: {e}")
    
    def detect_divisions(self):
        messagebox.showinfo("Coming Soon", "Cell division detection will be available soon!")
    
    def detect_apoptosis(self):
        messagebox.showinfo("Coming Soon", "Apoptosis detection will be available soon!")
    
    def analyze_morphology(self):
        messagebox.showinfo("Coming Soon", "Morphology analysis will be available soon!")
    
    def log(self, message):
        self.console.config(state='normal')
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)
        self.console.config(state='disabled')
        self.root.update()
    
    def run(self):
        self.log("Welcome to BioScience Engine Studio")
        self.log("Load images to begin analysis...")
        self.root.mainloop()


def main():
    app = CellTrackingGUI()
    app.run()


if __name__ == "__main__":
    main()