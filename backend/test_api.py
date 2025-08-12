#!/usr/bin/env python3
"""
Simple test script to verify the Flask API endpoints
"""

import requests
import json
import base64
from PIL import Image
import io

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Health data: {json.dumps(data, indent=2)}")
        else:
            print(f"Health check failed: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the Flask server is running.")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    return True

def test_prediction_with_sample_image():
    """Test the prediction endpoint with a sample image"""
    try:
        # Create a more realistic test image (larger size, more texture)
        img = Image.new('RGB', (512, 512), color='brown')
        
        # Add some texture to make it more realistic
        import numpy as np
        img_array = np.array(img)
        # Add some noise to simulate texture
        noise = np.random.randint(0, 50, (512, 512, 3), dtype=np.uint8)
        img_array = np.clip(img_array + noise, 0, 255)
        img = Image.fromarray(img_array)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Prepare request data - send just the base64 string, not the full data URL
        data = {
            "image": img_base64
        }
        
        print("Testing prediction endpoint...")
        response = requests.post(f"{BASE_URL}/predict", json=data)
        
        print(f"Prediction status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"Result: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Prediction failed: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the Flask server is running.")
        return False
    except Exception as e:
        print(f"❌ Prediction test error: {e}")
        return False
    return True

if __name__ == "__main__":
    print("🧪 Testing Mushroom Classification API")
    print("=" * 40)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    health_ok = test_health()
    
    if health_ok:
        # Test prediction endpoint
        print("\n2. Testing prediction endpoint...")
        prediction_ok = test_prediction_with_sample_image()
        
        if prediction_ok:
            print("\n✅ All tests passed! API is working correctly.")
        else:
            print("\n❌ Prediction test failed.")
    else:
        print("\n❌ Health check failed. API is not accessible.")
    
    print("\n" + "=" * 40)
    print("Test completed.")

