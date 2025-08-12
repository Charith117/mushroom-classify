#!/usr/bin/env python3
"""
Feature Comparison Tool
Compare training features with prediction features to identify mismatches
"""

import numpy as np
import pandas as pd
import joblib
import os
from debug_testing import extract_features_debug, columns

def analyze_training_data():
    """
    Analyze the training data structure to understand expected features
    """
    print("🔍 Analyzing Training Data Structure:")
    
    # Check if we have any training data files
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    # Look for any CSV or other data files that might contain training examples
    training_files = []
    for file in os.listdir(model_dir):
        if file.endswith(('.csv', '.pkl', '.joblib')) and 'train' in file.lower():
            training_files.append(file)
    
    if training_files:
        print(f"   Found potential training files: {training_files}")
        for file in training_files:
            file_path = os.path.join(model_dir, file)
            try:
                if file.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    print(f"   📊 {file}: {df.shape}, columns: {list(df.columns)}")
                elif file.endswith('.pkl'):
                    data = joblib.load(file_path)
                    if isinstance(data, pd.DataFrame):
                        print(f"   📊 {file}: {data.shape}, columns: {list(data.columns)}")
                    else:
                        print(f"   📊 {file}: {type(data)}, shape: {getattr(data, 'shape', 'N/A')}")
            except Exception as e:
                print(f"   ❌ Error reading {file}: {e}")
    else:
        print("   No training data files found in models directory")
    
    return training_files

def create_sample_training_features():
    """
    Create sample features that match what the model expects
    """
    print("\n🔧 Creating Sample Training Features:")
    
    # Create a sample feature vector with the expected structure
    sample_features = {
        'mean_r': 120.5,
        'mean_g': 110.3,
        'mean_b': 95.7,
        'std_r': 25.2,
        'std_g': 22.1,
        'std_b': 18.9,
        'mean_hue': 30.5,
        'mean_saturation': 45.2,
        'mean_value': 85.3,
        'contrast': 0.15,
        'correlation': 0.85,
        'energy': 0.12,
        'entropy': 4.25,
        'homogeneity': 0.78,
        'rms': 65.4,
        'smoothness': 0.45,
        'skewness': 0.12,
        'variance': 450.2,
        'kurtosis': 2.85
    }
    
    # Convert to list in the correct order
    sample_list = [sample_features[col] for col in columns]
    
    print(f"   Sample features created with {len(sample_list)} values")
    print(f"   Feature order matches columns: {columns}")
    
    return sample_list

def test_feature_consistency():
    """
    Test if the feature extraction produces consistent results
    """
    print("\n🧪 Testing Feature Consistency:")
    
    # Create multiple test images with slight variations
    base_img = np.random.randint(100, 150, (512, 512, 3), dtype=np.uint8)
    
    for i in range(3):
        # Add slight noise to create variation
        test_img = base_img + np.random.randint(-5, 5, (512, 512, 3), dtype=np.int16)
        test_img = np.clip(test_img, 0, 255).astype(np.uint8)
        
        print(f"   Test image {i+1}:")
        features = extract_features_debug(test_img)
        
        if features is not None:
            print(f"     Features extracted: {len(features)}")
            print(f"     Feature types: {[type(f).__name__ for f in features]}")
            print(f"     Sample values: {features[:5]}")
        else:
            print(f"     ❌ Feature extraction failed")

def compare_feature_ranges():
    """
    Compare expected feature ranges with actual extracted features
    """
    print("\n📊 Feature Range Analysis:")
    
    # Create test images with different characteristics
    test_images = [
        ("Dark image", np.random.randint(0, 50, (512, 512, 3), dtype=np.uint8)),
        ("Bright image", np.random.randint(200, 255, (512, 512, 3), dtype=np.uint8)),
        ("Colorful image", np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)),
        ("Gray image", np.random.randint(100, 150, (512, 512, 3), dtype=np.uint8))
    ]
    
    all_features = []
    
    for name, img in test_images:
        print(f"   Testing {name}:")
        features = extract_features_debug(img)
        
        if features is not None:
            all_features.append(features)
            print(f"     ✅ Features extracted successfully")
        else:
            print(f"     ❌ Feature extraction failed")
    
    if all_features:
        # Analyze feature ranges
        all_features = np.array(all_features)
        print(f"\n   📈 Feature Range Analysis:")
        print(f"     Total test images: {len(all_features)}")
        print(f"     Feature matrix shape: {all_features.shape}")
        
        for i, col in enumerate(columns):
            values = all_features[:, i]
            print(f"     {col:20s}: min={values.min():8.3f}, max={values.max():8.3f}, mean={values.mean():8.3f}")

def main():
    print("🔍 Mushroom Classification Feature Analysis Tool")
    print("=" * 60)
    
    # Analyze training data
    training_files = analyze_training_data()
    
    # Create sample features
    sample_features = create_sample_training_features()
    
    # Test feature consistency
    test_feature_consistency()
    
    # Compare feature ranges
    compare_feature_ranges()
    
    print("\n🏁 Feature analysis completed!")
    print("\n💡 Next steps:")
    print("   1. Run 'python debug_testing.py' for interactive testing")
    print("   2. Run 'python simple_uploader.py <image_path>' for single image testing")
    print("   3. Check if your training data columns match the expected columns")

if __name__ == "__main__":
    main()
