#!/usr/bin/env python3
"""
Simple GUI Tester for Mushroom Classification
Windows-compatible version with basic functionality
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
import joblib
import os
from PIL import Image, ImageTk
import threading

class SimpleMushroomTester:
    def __init__(self, root):
        self.root = root
        self.root.title("🍄 Mushroom Classifier Tester")
        self.root.geometry("900x800")
        self.root.configure(bg='#f0f0f0')
        
        # Load models
        self.load_models()
        
        # Variables
        self.current_image = None
        self.current_image_path = None
        
        # Create UI
        self.create_ui()
        
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
            
            # Define class names for better identification
            self.class_names = {
                0: "Abalone",
                1: "Pink Oyster", 
                2: "Boothan Oyster",
                3: "American Oyster"
            }
            
            print("✅ Models loaded successfully")
            print(f"✅ Class names loaded: {len(self.class_names)} classes")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {e}")
            self.root.destroy()
    
    def create_ui(self):
        """Create the user interface"""
        # Create main scrollable frame
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Main container
        main_container = ttk.Frame(scrollable_frame, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_container, text="🍄 Mushroom Classification Tester", 
                         font=('Arial', 18, 'bold'))
        title.pack(pady=(0, 20))
        
        # Image section
        image_frame = ttk.LabelFrame(main_container, text="Image Selection", padding="15")
        image_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Buttons row
        button_frame = ttk.Frame(image_frame)
        button_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(button_frame, text="📁 Select Image", 
                  command=self.select_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="🗑️ Clear", 
                  command=self.clear_image).pack(side=tk.LEFT)
        
        # Image display
        self.image_display = ttk.Label(image_frame, text="No image selected\n\nClick 'Select Image' to choose a file", 
                                      relief="solid", borderwidth=2, width=60)
        self.image_display.pack(pady=(10, 0))
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(main_container, text="Analysis", padding="15")
        analysis_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Analyze button
        self.analyze_btn = ttk.Button(analysis_frame, text="🔍 Analyze Image", 
                                     command=self.start_analysis, state="disabled")
        self.analyze_btn.pack(pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(analysis_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(analysis_frame, text="Ready to analyze", 
                                     font=('Arial', 10))
        self.status_label.pack()
        
        # Results section
        results_frame = ttk.LabelFrame(main_container, text="Results", padding="15")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, height=20, width=80, 
                                                     font=('Consolas', 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Pack the canvas and scrollbar
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel scrolling
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Initial message
        self.log("🚀 Mushroom Classification Tester Ready!")
        self.log("1. Click 'Select Image' to choose a mushroom image")
        self.log("2. Click 'Analyze Image' to run classification")
        self.log("3. View results below")
        self.log("")
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select Mushroom Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """Load and display the selected image"""
        try:
            # Load with OpenCV
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Error", "Could not load the image file")
                return
            
            # Store image
            self.current_image = img
            self.current_image_path = file_path
            
            # Get image info
            height, width = img.shape[:2]
            
            # Resize for display (max 400x400)
            max_size = 400
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_img = cv2.resize(img, (new_width, new_height))
            else:
                display_img = img
            
            # Convert BGR to RGB for PIL
            rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_img)
            
            # Update display
            self.image_display.configure(image=photo, text="")
            self.image_display.image = photo  # Keep reference
            
            # Enable analyze button
            self.analyze_btn.configure(state="normal")
            
            # Update status
            self.log(f"✅ Image loaded: {os.path.basename(file_path)}")
            self.log(f"   Dimensions: {width} x {height} pixels")
            self.log("   Ready for analysis!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            self.log(f"❌ Error loading image: {e}")
    
    def clear_image(self):
        """Clear the current image"""
        self.current_image = None
        self.current_image_path = None
        self.image_display.configure(image="", 
                                   text="No image selected\n\nClick 'Select Image' to choose a file")
        self.analyze_btn.configure(state="disabled")
        self.results_text.delete(1.0, tk.END)
        self.log("🗑️ Image cleared")
        self.log("Ready for new image")
    
    def start_analysis(self):
        """Start the image analysis"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        # Disable button and show progress
        self.analyze_btn.configure(state="disabled")
        self.progress.start()
        self.status_label.configure(text="Analyzing...")
        
        # Run in separate thread
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        """Run the analysis in background thread"""
        try:
            self.log("\n🔍 Starting analysis...")
            self.log("📊 Extracting features...")
            
            # Extract features using the same function from debug_testing
            features = self.extract_features(self.current_image)
            
            if features is None:
                self.log("❌ Feature extraction failed")
                return
            
            self.log(f"✅ Features extracted: {len(features)} features")
            self.log("🤖 Running model prediction...")
            
            # Make prediction
            prediction, confidence, probabilities = self.run_prediction(features)
            
            if prediction is None:
                self.log("❌ Model prediction failed")
                return
            
            # Display results
            self.log("\n🎯 ANALYSIS RESULTS:")
            self.log("=" * 50)
            
            # Get class name
            class_name = self.class_names.get(prediction, f"Unknown Class {prediction}")
            
            # Show prominent summary
            self.log("🏆 PREDICTION SUMMARY:")
            self.log("=" * 30)
            self.log(f"🍄 Mushroom Type: {class_name}")
            self.log(f"🎯 Confidence: {confidence:.1f}%")
            self.log(f"🔢 Class ID: {prediction}")
            self.log("")
            
            # Show all class probabilities
            self.log("📊 ALL CLASS PROBABILITIES:")
            self.log("-" * 40)
            for i, prob in enumerate(probabilities):
                class_name_i = self.class_names.get(i, f"Class {i}")
                prob_percent = prob * 100
                marker = "🏆" if i == prediction else "  "
                self.log(f"{marker} {i:2d}. {class_name_i:15s}: {prob_percent:6.2f}%")
            
            # Show feature summary
            self.log("\n📊 EXTRACTED FEATURES:")
            self.log("-" * 30)
            feature_names = [
                'mean_r', 'mean_g', 'mean_b',
                'std_r', 'std_g', 'std_b',
                'mean_hue', 'mean_saturation', 'mean_value',
                'contrast', 'correlation', 'energy', 'entropy', 'homogeneity',
                'rms', 'smoothness', 'skewness', 'variance', 'kurtosis'
            ]
            for i, (name, value) in enumerate(zip(feature_names, features)):
                self.log(f"{i+1:2d}. {name:20s}: {value:12.6f}")
            
            self.log("\n✅ Analysis completed successfully!")
            
        except Exception as e:
            self.log(f"❌ Analysis error: {e}")
        finally:
            # Re-enable UI
            self.root.after(0, self.analysis_completed)
    
    def extract_features(self, img_bgr):
        """Extract features from BGR image"""
        try:
            from skimage.feature import graycomatrix, graycoprops
            from scipy.stats import skew, kurtosis
            
            # Convert to grayscale
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # GLCM features
            glcm = graycomatrix(img_gray, distances=[1], angles=[0], symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            
            # Entropy
            glcm_norm = glcm.astype(np.float32) / glcm.sum()
            entropy = -np.sum(glcm_norm * np.log2(glcm_norm + 1e-10))
            
            # Color features
            mean_b, mean_g, mean_r = cv2.mean(img_bgr)[:3]
            std_b = np.std(img_bgr[:, :, 0])
            std_g = np.std(img_bgr[:, :, 1])
            std_r = np.std(img_bgr[:, :, 2])
            
            # HSV features
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            mean_hue = np.mean(hsv[:, :, 0])
            mean_sat = np.mean(hsv[:, :, 1])
            mean_val = np.mean(hsv[:, :, 2])
            
            # Statistical features
            gray_flat = img_gray.flatten().astype(np.float32)
            rms = np.sqrt(np.mean(np.square(gray_flat)))
            variance = np.var(gray_flat)
            smoothness = 1 - (1 / (1 + variance))
            skewness_val = skew(gray_flat)
            kurtosis_val = kurtosis(gray_flat)
            
            # Return features in correct order
            features = [
                float(mean_r), float(mean_g), float(mean_b),
                float(std_r), float(std_g), float(std_b),
                float(mean_hue), float(mean_sat), float(mean_val),
                float(contrast), float(correlation), float(energy), float(entropy), float(homogeneity),
                float(rms), float(smoothness), float(skewness_val), float(variance), float(kurtosis_val)
            ]
            
            return features
            
        except Exception as e:
            self.log(f"❌ Feature extraction error: {e}")
            return None
    
    def run_prediction(self, features):
        """Run model prediction"""
        try:
            import pandas as pd
            
            # Create DataFrame
            feature_names = [
                'mean_r', 'mean_g', 'mean_b',
                'std_r', 'std_g', 'std_b',
                'mean_hue', 'mean_saturation', 'mean_value',
                'contrast', 'correlation', 'energy', 'entropy', 'homogeneity',
                'rms', 'smoothness', 'skewness', 'variance', 'kurtosis'
            ]
            
            features_df = pd.DataFrame([features], columns=feature_names)
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            confidence = self.model.predict_proba(features_scaled).max() * 100
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            return prediction, confidence, probabilities
            
        except Exception as e:
            self.log(f"❌ Prediction error: {e}")
            return None, None, None
    
    def analysis_completed(self):
        """Called when analysis is completed"""
        self.analyze_btn.configure(state="normal")
        self.progress.stop()
        self.status_label.configure(text="Analysis completed")
    
    def log(self, message):
        """Add message to results area"""
        self.root.after(0, lambda: self.results_text.insert(tk.END, message + "\n"))
        self.root.after(0, lambda: self.results_text.see(tk.END))

def main():
    root = tk.Tk()
    app = SimpleMushroomTester(root)
    root.mainloop()

if __name__ == "__main__":
    main()
