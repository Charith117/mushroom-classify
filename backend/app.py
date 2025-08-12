import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from ultralytics import YOLO
import joblib
import os
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# === CONFIGURATION ===
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths - relative to current directory
model_dir = os.path.join(os.path.dirname(__file__), 'models')
model_path = os.path.join(model_dir, 'gb_best_model.joblib')
scaler_path = os.path.join(model_dir, 'scaler.joblib')
encoder_path = os.path.join(model_dir, 'label_encoder.joblib')
yolo_model_path = os.path.join(os.path.dirname(__file__), 'yolov5su.pt')

target_size = (512, 512)
padding_pixels = 10
confidence_threshold = 70.0  # Only accept predictions with >70% confidence
valid_extensions = ('.png', '.jpg', '.jpeg')
visualize_detection = False

# === Label Mapping ===
label_map = {
    'abalone': 0,
    'pink_oyster': 1,
    'buthan_oyster': 2,
    'american_oyster': 3
}
reverse_label_map = {v: k for k, v in label_map.items()}

# Scientific names for display
scientific_names = {
    'abalone': 'Pleurotus abalonus',
    'pink_oyster': 'Pleurotus djamor',
    'buthan_oyster': 'Pleurotus ostreatus',
    'american_oyster': 'Pleurotus ostreatus'
}

# Feature columns in the exact order expected by the model
columns = [
    'mean_r', 'mean_g', 'mean_b',
    'std_r', 'std_g', 'std_b',
    'mean_hue', 'mean_saturation', 'mean_value',
    'contrast', 'correlation', 'energy', 'entropy', 'homogeneity',
    'rms', 'smoothness', 'skewness', 'variance', 'kurtosis'
]

# Global variables for loaded models
model = None
scaler = None
label_encoder = None
yolo_model = None

# === Load Models ===
def load_models():
    """Load all ML models and components"""
    global model, scaler, label_encoder, yolo_model
    
    try:
        # Load ML model components
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        logger.info("ML models loaded successfully")
        
        # Load YOLO model with compatibility fixes
        try:
            yolo_model = YOLO(yolo_model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as yolo_error:
            logger.warning(f"Standard YOLO loading failed: {yolo_error}")
            logger.info("Attempting alternative YOLO loading method...")
            
            # Try alternative loading method for PyTorch compatibility
            try:
                import torch
                # Set weights_only to False for compatibility
                yolo_model = YOLO(yolo_model_path, task='detect')
                logger.info("YOLO model loaded with alternative method")
            except Exception as alt_error:
                logger.error(f"Alternative YOLO loading also failed: {alt_error}")
                logger.error("YOLO model could not be loaded. Server will start without object detection.")
                yolo_model = None
        
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

# === Entropy Function ===
def compute_entropy(glcm):
    glcm_norm = glcm.astype(np.float32) / glcm.sum()
    return -np.sum(glcm_norm * np.log2(glcm_norm + 1e-10))

# === Background Removal Utility ===
def remove_background(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    x1 = max(0, x1 - padding_pixels)
    y1 = max(0, y1 - padding_pixels)
    x2 = min(w, x2 + padding_pixels)
    y2 = min(h, y2 + padding_pixels)

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        raise ValueError("Bounding box too small for GrabCut")

    rect = (x1, y1, x2 - x1, y2 - y1)
    mask = np.zeros((h, w), np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        logger.warning("GrabCut failed, using rectangular mask.")
        mask[y1:y2, x1:x2] = 1

    final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    result = img * final_mask[:, :, np.newaxis]
    b, g, r = cv2.split(result)
    alpha = final_mask * 255
    return cv2.merge((b, g, r, alpha))  # RGBA

# === Contrast Enhancement Utility ===
def enhance_contrast(rgba_img):
    hsv = cv2.cvtColor(rgba_img[:, :, :3], cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))
    enhanced_rgb = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    b, g, r = cv2.split(enhanced_rgb)
    return cv2.merge((b, g, r, rgba_img[:, :, 3]))  # RGBA

# === Feature Extraction Function ===
def extract_features(img_bgr):
    """
    Extract features from a BGR image using the same logic as the standalone script.
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
        logging.error(f"Error extracting features: {e}")
        # Return default features if extraction fails
        return None

# === Image Validation Function ===
def is_likely_mushroom_image(img_bgr):
    """
    Basic validation to check if an image is likely a mushroom.
    Returns True if likely a mushroom, False otherwise.
    """
    try:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Check saturation - mushrooms typically have moderate saturation
        saturation = hsv[:, :, 1]
        mean_sat = np.mean(saturation)
        
        # Check brightness - mushrooms shouldn't be extremely bright or dark
        value = hsv[:, :, 2]
        mean_val = np.mean(value)
        
        # Check texture variation - mushrooms have some texture
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        texture_variation = np.std(gray)
        
        # Heuristic thresholds
        if mean_sat > 200:  # Too saturated (likely artificial)
            logging.info(f"Image too saturated: {mean_sat:.1f}")
            return False
        
        if mean_val > 220 or mean_val < 30:  # Too bright or too dark
            logging.info(f"Image brightness extreme: {mean_val:.1f}")
            return False
        
        if texture_variation < 15:  # Too smooth (likely not a mushroom)
            logging.info(f"Image too smooth, texture variation: {texture_variation:.1f}")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error in image validation: {e}")
        return True  # Default to allowing if validation fails

# === Main Prediction Logic ===
def predict_mushroom_from_image(img_bgr):
    """
    Predict mushroom type from a BGR image using the same processing pipeline as the standalone script.
    """
    try:
        # Apply median blur for noise reduction (same as standalone)
        img_filtered = cv2.medianBlur(img_bgr, 3)
        
        # Resize to target size (same as standalone)
        img_resized = cv2.resize(img_filtered, target_size)
        
        # YOLO detection
        if yolo_model is not None:
            try:
                results = yolo_model.predict(source=img_resized, conf=0.1, verbose=False)
                boxes = results[0].boxes
                
                if boxes is not None and boxes.xyxy.shape[0] > 0:
                    # Use detected bounding box
                    bbox_arr = boxes.xyxy.cpu().numpy()
                    bbox = bbox_arr[0]
                    logging.info(f"YOLO detected bbox: {bbox}")
                else:
                    # No detection, use full image
                    bbox = [0, 0, img_resized.shape[1], img_resized.shape[0]]
                    logging.info("No YOLO detection, using full image")
            except Exception as e:
                logging.warning(f"YOLO detection failed: {e}, using full image")
                bbox = [0, 0, img_resized.shape[1], img_resized.shape[0]]
        else:
            # YOLO not available, use full image
            bbox = [0, 0, img_resized.shape[1], img_resized.shape[0]]
            logging.info("YOLO model not available, using full image")
        
        # Background removal with padding (same as standalone)
        x1, y1, x2, y2 = map(int, bbox)
        h, w = img_resized.shape[:2]
        x1 = max(0, x1 - padding_pixels)
        y1 = max(0, y1 - padding_pixels)
        x2 = min(w, x2 + padding_pixels)
        y2 = min(h, y2 + padding_pixels)
        
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            logging.warning("Bounding box too small, using full image")
            bbox = [0, 0, w, h]
            x1, y1, x2, y2 = 0, 0, w, h
        
        # Apply GrabCut for background removal
        rect = (x1, y1, x2 - x1, y2 - y1)
        mask = np.zeros((h, w), np.uint8)
        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(img_resized, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            logging.warning("GrabCut failed, using rectangular mask")
            mask[y1:y2, x1:x2] = 1
        
        final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        result = img_resized * final_mask[:, :, np.newaxis]
        
        # Convert to BGR for feature extraction (same as standalone)
        enhanced_bgr = result[:, :, :3]  # Remove alpha channel
        
        # Extract features using the same function as standalone
        extracted_features = extract_features(enhanced_bgr)
        
        # Validate features
        if extracted_features is None or len(extracted_features) != 19:
            logging.error(f"Expected 19 features, got {len(extracted_features) if extracted_features is not None else 'None'}")
            return None, 0.0, -1
        
        # Convert to DataFrame and scale
        features_df = pd.DataFrame([extracted_features], columns=columns)
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled).max() * 100
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            logging.info(f"Prediction confidence {confidence:.2f}% below threshold {confidence_threshold}%")
            return None, confidence, -1
        
        # Check if image is likely a mushroom
        if not is_likely_mushroom_image(img_bgr):
            logging.info("Image validation failed - likely not a mushroom")
            return None, confidence, -2
        
        return prediction, confidence, prediction
        
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return None, 0.0, -1

# === API Endpoints ===
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'ml_classifier': model is not None,
            'scaler': scaler is not None,
            'label_encoder': label_encoder is not None,
            'yolo_detector': yolo_model is not None
        },
        'yolo_status': 'loaded' if yolo_model is not None else 'failed_to_load',
        'message': 'Server is running. YOLO detection may be limited if model failed to load.'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Get image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_base64 = data['image']
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # OpenCV returns BGR format, which is what we need for processing
        # No need to convert RGB to BGR since imdecode already gives us BGR
        
        # Make prediction
        prediction, confidence, class_id = predict_mushroom_from_image(img_bgr)
        
        # Check if prediction is uncertain
        if class_id == -1:
            return jsonify({
                'error': 'Uncertain prediction',
                'message': f'Prediction confidence ({confidence:.1f}%) is too low. This might not be a mushroom or the image quality is poor.',
                'confidence': confidence,
                'suggestion': 'Please upload a clearer image of a mushroom, or this might not be a mushroom image.'
            })
        
        # Check if image is not a mushroom
        if class_id == -2:
            return jsonify({
                'error': 'Not a mushroom image',
                'message': 'This image appears to be something else.',
                'confidence': confidence,
                'suggestion': 'Please upload an image of a mushroom. This image appears to be something else.'
            })
        
        # Check if prediction failed completely
        if prediction is None:
            return jsonify({
                'error': 'Prediction failed',
                'message': 'Could not process the image for prediction.',
                'suggestion': 'Please try with a different image or check the image format.'
            }), 500
        
        # Extract features for response (only if prediction was successful)
        response_features = extract_features(img_bgr)
        if response_features is None or len(response_features) != 19:
            return jsonify({
                'error': 'Feature extraction failed',
                'message': 'Could not extract features from the image.',
                'suggestion': 'Please try with a different image.'
            }), 500
        
        # Format response - ensure all values are native Python types for JSON serialization
        response = {
            'mushroomType': {
                'common': reverse_label_map.get(int(prediction), f"Unknown class ({int(prediction)})").replace('_', ' ').title(),
                'scientific': scientific_names.get(reverse_label_map.get(int(prediction), f"Unknown class ({int(prediction)})"), 'Unknown')
            },
            'confidence': float(confidence),
            'classId': int(class_id),
            'features': {
                'colorMean': [float(x) for x in response_features[:3]],  # RGB means - convert to native Python float
                'textureFeatures': {
                    'contrast': round(float(response_features[9]), 3),      # contrast
                    'correlation': round(float(response_features[10]), 3),  # correlation
                    'energy': round(float(response_features[11]), 3),       # energy
                    'homogeneity': round(float(response_features[13]), 3),  # homogeneity
                    'entropy': round(float(response_features[12]), 3)      # entropy
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# === Error Handlers ===
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# === Main Application ===
if __name__ == '__main__':
    # Load models before starting server
    if not load_models():
        logging.error("Failed to load models. Exiting.")
        exit(1)
    
    logging.info("Starting Flask server...")
    logging.info("API will be available at http://localhost:8000")
    logging.info("Health check: http://localhost:8000/health")
    logging.info("Prediction endpoint: http://localhost:8000/predict")
    
    app.run(host='0.0.0.0', port=8000, debug=False)