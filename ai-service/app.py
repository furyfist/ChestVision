import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageStat 
import torch
import io 
import torchvision.transforms as transforms
from torchvision import models

# Configuration & Model Loading

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'lung_cancer_classifier.pth')

CLASS_NAMES = [
    'adenocarcinoma', 
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 
    'large.cell.carcinoma', 
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 
    'normal', 
    'squamous.cell.carcinoma', 
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
] 

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture (ResNet-50)
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))

# Load the trained weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"--- Model loaded successfully on {device} ---")
except Exception as e:
    print(f"Error loading model: {e}")

# Helper Functions 

def check_if_ct_scan(pil_image):
    """
    Returns False if image has high color saturation (likely a photo/object).
    Returns True if image is mostly grayscale (likely a CT scan/X-ray).
    """
    try:
        hsv_img = pil_image.convert('HSV')
        # Get average saturation (Index 1 in HSV)
        mean_saturation = ImageStat.Stat(hsv_img).mean[1]
        
        # Threshold: Real CT scans usually have saturation < 5. 
        # We set 25 to be safe (allows some compression artifacts).
        if mean_saturation > 25: 
            return False
        return True
    except Exception:
        return True # Default to True if check fails to avoid blocking valid images

def transform_image(pil_image):
    """
    Accepts a PIL Image object and converts it to a tensor.
    """
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Ensure RGB (ResNet expects 3 channels)
    img_rgb = pil_image.convert('RGB')
    return transform_pipeline(img_rgb).unsqueeze(0) 


# Flask App Definition

app = Flask(__name__)
CORS(app) 

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400

    file = request.files['file']
    if not file:
        return jsonify({'error': 'no file selected'}), 400

    try:
        # 1. Open the image directly from bytes
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))

        # 2. The Gatekeeper Check
        if not check_if_ct_scan(pil_img):
            print("Block: Image detected as non-CT scan (High Saturation)")
            return jsonify({
                'prediction': 'Invalid Image',
                'confidence': 0.0,
                'invalid_image': True, 
                'message': 'Upload rejected: Image appears to be colorful. Please upload a grayscale CT Scan.'
            })

        # 3. Transform and Predict
        tensor = transform_image(pil_img).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            confidence_score = confidence.item() * 100
            predicted_class = CLASS_NAMES[predicted_idx.item()]

        print(f"Prediction: {predicted_class} (confidence: {confidence_score:.1f}%)")
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence_score, 1),
            'invalid_image': False
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'error during prediction'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)