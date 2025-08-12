#!/usr/bin/env python3
"""
Comprehensive Testing Environment for Mushroom Classification
This script helps debug feature extraction and model prediction issues
"""

import cv2
import numpy as np
import pandas as pd
import joblib
import os
import base64
from PIL import Image
import io
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# Model paths
model_dir = os.path.join(os.path.dirname(__file__), 'models')
model_path = os.path.join(model_dir, 'gb_best_model.joblib')
scaler_path = os.path.join(model_dir, 'scaler.joblib')
encoder_path = os.path.join(model_dir, 'label_encoder.joblib')

# Load models
print("🔍 Loading models...")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)
print("✅ Models loaded successfully")

# Class names for better identification
class_names = {
    0: "Abalone",
    1: "Pink Oyster", 
    2: "Boothan Oyster",
    3: "American Oyster"
}
print(f"✅ Class names loaded: {len(class_names)} classes")

# Feature columns in the exact order expected by the model
columns = [
    'mean_r', 'mean_g', 'mean_b',
    'std_r', 'std_g', 'std_b',
    'mean_hue', 'mean_saturation', 'mean_value',
    'contrast', 'correlation', 'energy', 'entropy', 'homogeneity',
    'rms', 'smoothness', 'skewness', 'variance', 'kurtosis'
]

print(f"📊 Expected features: {len(columns)}")
print(f"📋 Feature names: {columns}")

def compute_entropy(glcm):
    """Compute entropy from GLCM"""
    glcm_norm = glcm.astype(np.float32) / glcm.sum()
    return -np.sum(glcm_norm * np.log2(glcm_norm + 1e-10))

def extract_features_debug(img_bgr):
    """
    Extract features with detailed debugging information
    """
    print("\n🔍 Feature Extraction Debug:")
    print(f"   Input image shape: {img_bgr.shape}")
    print(f"   Input image dtype: {img_bgr.dtype}")
    print(f"   Input image range: {img_bgr.min()} to {img_bgr.max()}")
    
    try:
        # Convert BGR to grayscale for texture analysis
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        print(f"   Grayscale shape: {img_gray.shape}")
        print(f"   Grayscale range: {img_gray.min()} to {img_gray.max()}")
        
        # GLCM texture features
        print("   Computing GLCM features...")
        glcm = graycomatrix(img_gray, distances=[1], angles=[0], symmetric=True, normed=True)
        print(f"   GLCM shape: {glcm.shape}")
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        entropy = compute_entropy(glcm)
        
        print(f"   GLCM features computed:")
        print(f"     Contrast: {contrast:.6f}")
        print(f"     Correlation: {correlation:.6f}")
        print(f"     Energy: {energy:.6f}")
        print(f"     Homogeneity: {homogeneity:.6f}")
        print(f"     Entropy: {entropy:.6f}")
        
        # Color statistics (BGR order as expected)
        print("   Computing color features...")
        mean_b, mean_g, mean_r = cv2.mean(img_bgr)[:3]
        std_b = np.std(img_bgr[:, :, 0])
        std_g = np.std(img_bgr[:, :, 1])
        std_r = np.std(img_bgr[:, :, 2])
        
        print(f"   Color means (BGR): {mean_b:.2f}, {mean_g:.2f}, {mean_r:.2f}")
        print(f"   Color stds (BGR): {std_b:.2f}, {std_g:.2f}, {std_r:.2f}")
        
        # HSV color features
        print("   Computing HSV features...")
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsv[:, :, 0])
        mean_sat = np.mean(hsv[:, :, 1])
        mean_val = np.mean(hsv[:, :, 2])
        
        print(f"   HSV means: {mean_hue:.2f}, {mean_sat:.2f}, {mean_val:.2f}")
        
        # Statistical texture features
        print("   Computing statistical features...")
        gray_flat = img_gray.flatten().astype(np.float32)
        rms = np.sqrt(np.mean(np.square(gray_flat)))
        variance = np.var(gray_flat)
        smoothness = 1 - (1 / (1 + variance))
        skewness_val = skew(gray_flat)
        kurtosis_val = kurtosis(gray_flat)
        
        print(f"   Statistical features:")
        print(f"     RMS: {rms:.2f}")
        print(f"     Variance: {variance:.2f}")
        print(f"     Smoothness: {smoothness:.6f}")
        print(f"     Skewness: {skewness_val:.6f}")
        print(f"     Kurtosis: {kurtosis_val:.6f}")
        
        # Return features in the exact order expected by the model
        features = [
            float(mean_r), float(mean_g), float(mean_b),      # RGB means
            float(std_r), float(std_g), float(std_b),          # RGB stds
            float(mean_hue), float(mean_sat), float(mean_val), # HSV means
            float(contrast), float(correlation), float(energy), float(entropy), float(homogeneity),  # GLCM features
            float(rms), float(smoothness), float(skewness_val), float(variance), float(kurtosis_val)  # Statistical features
        ]
        
        print(f"   ✅ Total features extracted: {len(features)}")
        print(f"   📊 Feature types: {[type(f).__name__ for f in features]}")
        
        return features
        
    except Exception as e:
        print(f"   ❌ Error extracting features: {e}")
        return None

def test_model_prediction(features):
    """
    Test model prediction with extracted features
    """
    print("\n🤖 Model Prediction Debug:")
    
    try:
        # Convert to DataFrame
        features_df = pd.DataFrame([features], columns=columns)
        print(f"   Features DataFrame shape: {features_df.shape}")
        print(f"   Features DataFrame columns: {list(features_df.columns)}")
        
        # Check for any NaN or infinite values
        print(f"   NaN values: {features_df.isna().sum().sum()}")
        print(f"   Infinite values: {np.isinf(features_df.values).sum()}")
        
        # Scale features
        print("   Scaling features...")
        features_scaled = scaler.transform(features_df)
        print(f"   Scaled features shape: {features_scaled.shape}")
        print(f"   Scaled features range: {features_scaled.min():.6f} to {features_scaled.max():.6f}")
        
        # Make prediction
        print("   Making prediction...")
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled).max() * 100
        
        print(f"   Raw prediction: {prediction}")
        print(f"   Prediction type: {type(prediction)}")
        print(f"   Confidence: {confidence:.2f}%")
        
        # Get class probabilities
        probabilities = model.predict_proba(features_scaled)[0]
        print(f"   Class probabilities: {probabilities}")
        
        return prediction, confidence, probabilities
        
    except Exception as e:
        print(f"   ❌ Error in model prediction: {e}")
        return None, None, None

def test_with_sample_image():
    """
    Test with a sample image to verify the pipeline
    """
    print("\n🧪 Testing with Sample Image:")
    
    # Create a test image (brown with texture to simulate mushroom)
    img = np.random.randint(100, 150, (512, 512, 3), dtype=np.uint8)
    img[:, :, 0] = np.random.randint(80, 120, (512, 512), dtype=np.uint8)  # Blue channel
    img[:, :, 1] = np.random.randint(100, 140, (512, 512), dtype=np.uint8)  # Green channel  
    img[:, :, 2] = np.random.randint(120, 160, (512, 512), dtype=np.uint8)  # Red channel
    
    print(f"   Created test image: {img.shape}, dtype: {img.dtype}")
    
    # Extract features
    features = extract_features_debug(img)
    
    if features is not None:
        # Test model prediction
        prediction, confidence, probabilities = test_model_prediction(features)
        
        if prediction is not None:
            print(f"\n🎯 Test Results:")
            class_name = class_names.get(prediction, f"Unknown Class {prediction}")
            print(f"   Prediction: {prediction} ({class_name})")
            print(f"   Confidence: {confidence:.2f}%")
            print(f"   Probabilities: {probabilities}")
            
            # Show all class probabilities with names
            print(f"\n📊 All Class Probabilities:")
            for i, prob in enumerate(probabilities):
                class_name_i = class_names.get(i, f"Class {i}")
                prob_percent = prob * 100
                marker = "🏆" if i == prediction else "  "
                print(f"   {marker} {i}: {class_name_i:20s} - {prob_percent:6.2f}%")
            
            return True
    
    return False

def interactive_image_upload():
    """
    Interactive image upload and testing
    """
    print("\n📸 Interactive Image Testing:")
    print("   Please provide an image path or type 'sample' for test image:")
    
    while True:
        user_input = input("   Image path (or 'sample'/'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'sample':
            test_with_sample_image()
            continue
        
        # Check if file exists
        if not os.path.exists(user_input):
            print(f"   ❌ File not found: {user_input}")
            continue
        
        try:
            # Load image
            print(f"   📁 Loading image: {user_input}")
            img = cv2.imread(user_input)
            
            if img is None:
                print(f"   ❌ Could not load image: {user_input}")
                continue
            
            print(f"   ✅ Image loaded: {img.shape}")
            
            # Extract features
            features = extract_features_debug(img)
            
            if features is not None:
                # Test model prediction
                prediction, confidence, probabilities = test_model_prediction(features)
                
                if prediction is not None:
                    print(f"\n🎯 Results for {user_input}:")
                    print(f"   Prediction: {prediction}")
                    print(f"   Confidence: {confidence:.2f}%")
                    print(f"   Probabilities: {probabilities}")
                    
                    # Show feature summary
                    print(f"\n📊 Feature Summary:")
                    for i, (name, value) in enumerate(zip(columns, features)):
                        print(f"   {name}: {value:.6f}")
            
        except Exception as e:
            print(f"   ❌ Error processing image: {e}")

if __name__ == "__main__":
    print("🍄 Mushroom Classification Debug Tool")
    print("=" * 50)
    
    # Test with sample image first
    print("\n1️⃣ Testing with sample image...")
    sample_test_ok = test_with_sample_image()
    
    if sample_test_ok:
        print("✅ Sample test completed successfully")
    else:
        print("❌ Sample test failed")
    
    # Interactive testing
    print("\n2️⃣ Interactive image testing...")
    interactive_image_upload()
    
    print("\n🏁 Debug session completed!")
