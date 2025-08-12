# Mushroom Identification Backend

This Flask API provides mushroom identification using machine learning models and YOLO object detection.

## Features

- **ML Model**: Gradient Boosting classifier for mushroom classification
- **Object Detection**: YOLOv5 for mushroom detection in images
- **Feature Extraction**: Advanced texture and color feature analysis
- **Background Removal**: Automatic background removal using GrabCut
- **REST API**: Flask-based API with CORS support

## Setup Instructions

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify Model Files:**
Ensure these files are present in the backend directory:
- `models/gb_best_model.joblib` - ML classifier
- `models/scaler.joblib` - Feature scaler
- `models/label_encoder.joblib` - Label encoder
- `yolov5su.pt` - YOLO model

3. **Run the API:**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### GET /health
Health check endpoint to verify API status and model loading.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### POST /predict
Main prediction endpoint for mushroom identification.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
}
```

**Response:**
```json
{
  "mushroomType": {
    "common": "Pink Oyster",
    "scientific": "Pleurotus djamor"
  },
  "confidence": 94.7,
  "classId": 1,
  "features": {
    "colorMean": [145.2, 123.8, 98.5],
    "textureFeatures": {
      "contrast": 0.234,
      "correlation": 0.876,
      "energy": 0.145,
      "homogeneity": 0.678,
      "entropy": 4.321
    }
  }
}
```

## Supported Mushroom Classes

1. **Abalone** (Pleurotus abalonus) - Class ID: 0
2. **Pink Oyster** (Pleurotus djamor) - Class ID: 1
3. **Bhutan Oyster** (Pleurotus ostreatus) - Class ID: 2
4. **American Oyster** (Pleurotus ostreatus) - Class ID: 3

## Testing

Run the test script to verify API functionality:
```bash
python test_api.py
```

## Integration with Frontend

The frontend is configured to call this API at `http://localhost:5000/predict`. 
Make sure the backend is running before testing the frontend.

## Error Handling

The API includes comprehensive error handling:
- Invalid image data
- Model loading failures
- Prediction errors
- Proper HTTP status codes

## Logging

The API includes detailed logging for debugging:
- Model loading status
- Prediction requests
- Error details
- Server startup information

## Troubleshooting

1. **Models not loading**: Check file paths and ensure all model files are present
2. **YOLO errors**: Verify `yolov5su.pt` is in the correct location
3. **Memory issues**: The YOLO model requires significant RAM (~2-4GB)
4. **Port conflicts**: Ensure port 5000 is available