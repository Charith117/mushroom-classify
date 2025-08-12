#!/usr/bin/env python3
"""
Simple Image Uploader and Tester for Mushroom Classification
"""

import cv2
import numpy as np
import joblib
import os
import argparse
from debug_testing import extract_features_debug, test_model_prediction, columns

def load_models():
    """Load the trained models"""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(model_dir, 'gb_best_model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    encoder_path = os.path.join(model_dir, 'label_encoder.joblib')
    
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
    
    return model, scaler, label_encoder

def test_single_image(image_path, model, scaler, label_encoder):
    """Test a single image with detailed output"""
    print(f"\n🍄 Testing image: {image_path}")
    print("=" * 60)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return False
    
    print(f"📁 Image loaded successfully:")
    print(f"   Shape: {img.shape}")
    print(f"   Data type: {img.dtype}")
    print(f"   Value range: {img.min()} to {img.max()}")
    
    # Extract features
    features = extract_features_debug(img)
    
    if features is None:
        print("❌ Feature extraction failed")
        return False
    
    # Test model prediction
    prediction, confidence, probabilities = test_model_prediction(features)
    
    if prediction is None:
        print("❌ Model prediction failed")
        return False
    
    # Display results
    print(f"\n🎯 PREDICTION RESULTS:")
    class_name = class_names.get(prediction, f"Unknown Class {prediction}")
    print(f"   🍄 Mushroom Type: {class_name}")
    print(f"   🎯 Confidence: {confidence:.2f}%")
    print(f"   🔢 Class ID: {prediction}")
    
    # Show all class probabilities with names
    print(f"\n📊 ALL CLASS PROBABILITIES:")
    for i, prob in enumerate(probabilities):
        class_name_i = class_names.get(i, f"Class {i}")
        prob_percent = prob * 100
        marker = "🏆" if i == prediction else "  "
        print(f"   {marker} {i}: {class_name_i:20s} - {prob_percent:6.2f}%")
    
    # Show feature values
    print(f"\n📊 EXTRACTED FEATURES:")
    for i, (name, value) in enumerate(zip(columns, features)):
        print(f"   {i+1:2d}. {name:20s}: {value:12.6f}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test mushroom classification on a single image')
    parser.add_argument('image_path', help='Path to the image file to test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"❌ Image file not found: {args.image_path}")
        return
    
    try:
        # Load models
        model, scaler, label_encoder = load_models()
        
        # Test the image
        success = test_single_image(args.image_path, model, scaler, label_encoder)
        
        if success:
            print(f"\n✅ Image testing completed successfully!")
        else:
            print(f"\n❌ Image testing failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
