# ai-service/app.py

import os
from flask import Flask, request, jsonify
from PIL import Image
import torch
import io 
import torchvision.transforms as transforms
from torchvision import models

# --- 1. Configuration & Model Loading ---

# Define the path to the model relative to this script's location
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'lung_cancer_classifier.pth')

# Define class names from your dataset
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
# Make sure the architecture matches exactly what you used for training
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))

# Load the trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode (very important!)

print(f"--- Model loaded successfully on {device} ---")


# --- 2. Image Transformation ---

# Define the same transformations you used during validation/testing
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0) # Add batch dimension


# --- 3. Flask App Definition ---

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400

    file = request.files['file']
    if not file:
        return jsonify({'error': 'no file selected'}), 400

    try:
        # Get image bytes from the request
        img_bytes = file.read()

        # Transform the image and move to the correct device
        tensor = transform_image(img_bytes).to(device)

        # Make a prediction
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = CLASS_NAMES[predicted_idx.item()]

        print(f"Prediction successful: {predicted_class}")
        return jsonify({'prediction': predicted_class})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'error during prediction'}), 500

# This block allows you to run the app directly with `python app.py` for testing
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)