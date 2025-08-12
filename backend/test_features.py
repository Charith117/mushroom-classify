#!/usr/bin/env python3
"""
Test script to verify feature extraction is working correctly.
This will help debug any issues with the feature extraction pipeline.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
target_size = (512, 512)

def compute_entropy(glcm):
    """Compute entropy from GLCM matrix"""
    glcm_norm = glcm.astype(np.float32) / glcm.sum()
    return -np.sum(glcm_norm * np.log2(glcm_norm + 1e-10))

def extract_features(img_bgr):
    """
    Extract features from a BGR image using the same logic as the main app.
    Returns a list of 19 features in the exact order expected by the model.
    """
    try:
        # Convert BGR to grayscale for texture analysis
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # GLCM texture features
        glcm = graycomatrix(img_gray, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        entropy = compute_entropy(glcm)
        
        # Color statistics (BGR order as expected)
        mean_b, mean_g, mean_r = cv2.mean(img_bgr)[:3]
        std_b = np.std(img_bgr[:, :, 0])
        std_g = np.std(img_bgr[:, :, 1])
        std_r = np.std(img_bgr[:, :, 2])
        
        # HSV color features
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsv[:, :, 0])
        mean_sat = np.mean(hsv[:, :, 1])
        mean_val = np.mean(hsv[:, :, 2])
        
        # Statistical texture features
        gray_flat = img_gray.flatten().astype(np.float32)
        rms = np.sqrt(np.mean(np.square(gray_flat)))
        variance = np.var(gray_flat)
        smoothness = 1 - (1 / (1 + variance))
        skewness_val = skew(gray_flat)
        kurtosis_val = kurtosis(gray_flat)
        
        # Return features in the exact order expected by the model
        features = [
            float(mean_r), float(mean_g), float(mean_b),      # RGB means
            float(std_r), float(std_g), float(std_b),          # RGB stds
            float(mean_hue), float(mean_sat), float(mean_val), # HSV means
            float(contrast), float(correlation), float(energy), float(entropy), float(homogeneity),  # GLCM features
            float(rms), float(smoothness), float(skewness_val), float(variance), float(kurtosis_val)  # Statistical features
        ]
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def test_feature_extraction():
    """Test feature extraction with a simple test image"""
    
    # Create a simple test image (512x512 with some variation)
    test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Add some texture
    test_img[:, :, 0] = np.random.randint(100, 200, (512, 512), dtype=np.uint8)  # Blue channel
    test_img[:, :, 1] = np.random.randint(80, 180, (512, 512), dtype=np.uint8)   # Green channel  
    test_img[:, :, 2] = np.random.randint(120, 220, (512, 512), dtype=np.uint8)  # Red channel
    
    logger.info(f"Test image shape: {test_img.shape}")
    logger.info(f"Test image dtype: {test_img.dtype}")
    
    # Extract features
    features = extract_features(test_img)
    
    if features is None:
        logger.error("Feature extraction failed!")
        return False
    
    logger.info(f"Successfully extracted {len(features)} features")
    
    # Feature names for reference
    feature_names = [
        'mean_r', 'mean_g', 'mean_b',
        'std_r', 'std_g', 'std_b', 
        'mean_hue', 'mean_sat', 'mean_val',
        'contrast', 'correlation', 'energy', 'entropy', 'homogeneity',
        'rms', 'smoothness', 'skewness', 'variance', 'kurtosis'
    ]
    
    # Print feature values
    logger.info("Feature values:")
    for i, (name, value) in enumerate(zip(feature_names, features)):
        logger.info(f"  {i:2d}. {name:15s}: {value:10.4f}")
    
    # Validate feature types
    all_float = all(isinstance(f, float) for f in features)
    logger.info(f"All features are float: {all_float}")
    
    # Validate feature ranges (basic sanity checks)
    rgb_means = features[:3]
    rgb_stds = features[3:6]
    hsv_means = features[6:9]
    
    logger.info(f"RGB means: {rgb_means}")
    logger.info(f"RGB stds: {rgb_stds}")
    logger.info(f"HSV means: {hsv_means}")
    
    # Check if values are reasonable
    if all(0 <= m <= 255 for m in rgb_means):
        logger.info("✓ RGB means are in valid range [0, 255]")
    else:
        logger.warning("⚠ RGB means are outside valid range [0, 255]")
    
    if all(0 <= s <= 255 for s in rgb_stds):
        logger.info("✓ RGB stds are in valid range [0, 255]")
    else:
        logger.warning("⚠ RGB stds are outside valid range [0, 255]")
    
    if all(0 <= h <= 179 for h in hsv_means[:1]) and all(0 <= v <= 255 for v in hsv_means[1:]):
        logger.info("✓ HSV values are in valid ranges")
    else:
        logger.warning("⚠ HSV values are outside valid ranges")
    
    return True

if __name__ == "__main__":
    logger.info("Testing feature extraction...")
    success = test_feature_extraction()
    
    if success:
        logger.info("✅ Feature extraction test passed!")
    else:
        logger.error("❌ Feature extraction test failed!")
