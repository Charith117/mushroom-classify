#!/usr/bin/env python3
"""
GUI Tester for Mushroom Classification
A simple interface to upload and test images
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import joblib
import os
from PIL import Image, ImageTk
import threading
from debug_testing import extract_features_debug, test_model_prediction, columns

class MushroomClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🍄 Mushroom Classification Tester")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Load models
        self.load_models()
        
        # Create UI
        self.create_widgets()
        
        # Variables
        self.current_image = None
        self.current_image_path = None
        
    def load_models(self):
        """Load the trained models"""
        try:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            model_path = os.path.join(model_dir, 'gb_best_model.joblib')
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            encoder_path = os.path.join(model_dir, 'label_encoder.joblib')
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            
            print("✅ Models loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {e}")
            self.root.destroy()
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🍄 Mushroom Classification Tester", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Image upload section
        upload_frame = ttk.LabelFrame(main_frame, text="Image Upload", padding="10")
        upload_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Upload buttons
        ttk.Button(upload_frame, text="📁 Browse Image", 
                  command=self.browse_image).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(upload_frame, text="🗑️ Clear Image", 
                  command=self.clear_image).grid(row=0, column=1)
        
        # Drag and drop label
        drop_label = ttk.Label(upload_frame, text="Or drag and drop an image here", 
                              font=('Arial', 10, 'italic'))
        drop_label.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        # Image display
        self.image_label = ttk.Label(upload_frame, text="No image selected", 
                                    relief="solid", borderwidth=2, width=50)
        self.image_label.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(main_frame, text="Analysis", padding="10")
        analysis_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Analyze button
        self.analyze_button = ttk.Button(analysis_frame, text="🔍 Analyze Image", 
                                        command=self.analyze_image, state="disabled")
        self.analyze_button.grid(row=0, column=0, pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(analysis_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Results text
        self.results_text = tk.Text(results_frame, height=15, width=80, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        upload_frame.columnconfigure(0, weight=1)
        analysis_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Bind drag and drop events
        self.image_label.bind('<Drop>', self.on_drop)
        self.image_label.bind('<DragEnter>', self.on_drag_enter)
        self.image_label.bind('<DragLeave>', self.on_drag_leave)
        
    def browse_image(self):
        """Browse for an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """Load and display an image"""
        try:
            # Load image with OpenCV
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Error", "Could not load the image")
                return
            
            # Store the image and path
            self.current_image = img
            self.current_image_path = file_path
            
            # Resize for display (max 300x300)
            height, width = img.shape[:2]
            max_size = 300
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img_display = cv2.resize(img, (new_width, new_height))
            else:
                img_display = img
            
            # Convert BGR to RGB for PIL
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update display
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            # Enable analyze button
            self.analyze_button.configure(state="normal")
            
            # Update status
            self.log_message(f"✅ Image loaded: {os.path.basename(file_path)}")
            self.log_message(f"   Size: {width}x{height} pixels")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def clear_image(self):
        """Clear the current image"""
        self.current_image = None
        self.current_image_path = None
        self.image_label.configure(image="", text="No image selected")
        self.analyze_button.configure(state="disabled")
        self.results_text.delete(1.0, tk.END)
        self.log_message("🗑️ Image cleared")
    
    def analyze_image(self):
        """Analyze the current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        # Disable button and show progress
        self.analyze_button.configure(state="disabled")
        self.progress.start()
        
        # Run analysis in separate thread
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        """Run the analysis in a separate thread"""
        try:
            self.log_message("\n🔍 Starting analysis...")
            
            # Extract features
            self.log_message("📊 Extracting features...")
            features = extract_features_debug(self.current_image)
            
            if features is None:
                self.log_message("❌ Feature extraction failed")
                return
            
            self.log_message(f"✅ Features extracted: {len(features)} features")
            
            # Test model prediction
            self.log_message("🤖 Running model prediction...")
            prediction, confidence, probabilities = test_model_prediction(features)
            
            if prediction is None:
                self.log_message("❌ Model prediction failed")
                return
            
            # Display results
            self.log_message("\n🎯 ANALYSIS RESULTS:")
            self.log_message("=" * 50)
            self.log_message(f"Class ID: {prediction}")
            self.log_message(f"Confidence: {confidence:.2f}%")
            self.log_message(f"All Probabilities: {probabilities}")
            
            # Show feature summary
            self.log_message("\n📊 EXTRACTED FEATURES:")
            self.log_message("-" * 30)
            for i, (name, value) in enumerate(zip(columns, features)):
                self.log_message(f"{i+1:2d}. {name:20s}: {value:12.6f}")
            
            self.log_message("\n✅ Analysis completed successfully!")
            
        except Exception as e:
            self.log_message(f"❌ Analysis error: {e}")
        finally:
            # Re-enable button and stop progress
            self.root.after(0, self.analysis_completed)
    
    def analysis_completed(self):
        """Called when analysis is completed"""
        self.analyze_button.configure(state="normal")
        self.progress.stop()
    
    def log_message(self, message):
        """Add a message to the results text"""
        self.root.after(0, lambda: self.results_text.insert(tk.END, message + "\n"))
        self.root.after(0, lambda: self.results_text.see(tk.END))
    
    def on_drop(self, event):
        """Handle file drop event"""
        file_path = event.data
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            self.load_image(file_path)
        else:
            messagebox.showwarning("Warning", "Please drop an image file")
    
    def on_drag_enter(self, event):
        """Handle drag enter event"""
        self.image_label.configure(relief="solid", borderwidth=3)
    
    def on_drag_leave(self, event):
        """Handle drag leave event"""
        self.image_label.configure(relief="solid", borderwidth=2)

def main():
    root = tk.Tk()
    app = MushroomClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
