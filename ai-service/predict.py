import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

# model setup
print("Setting up for predictions......")

device = torch.device("cude" if torch.cuda.is_available() else "cpu")

# all classes
class_names = [
    'adenocarcinoma', 
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 
    'large.cell.carcinoma', 
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 
    'normal', 
    'squamous.cell.carcinoma', 
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
]

# creating same model architecture as in training
model = models.resnet50() # No weights argument here
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

# Loading the Saved Model Weights 
model_path = "lung_cancer_classifier.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval() # Set model to evaluation mode
print(f"Model loaded from {model_path} and set to evaluation mode.")

# Image Transformation Pipeline
# This must be IDENTICAL to the transformations used for validation/training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. Load and Predict a Single Image ---
def predict_image(image_path):
    try:
        # Open the image file
        img = Image.open(image_path).convert('RGB')
        
        # Apply the transformations
        img_t = transform(img)
        
        # The model expects a batch of images, so we add a batch dimension
        # (e.g., from [3, 224, 224] to [1, 3, 224, 224])
        batch_t = torch.unsqueeze(img_t, 0)
        batch_t = batch_t.to(device)
        
        # Make the prediction
        with torch.no_grad():
            outputs = model(batch_t)
        
        # Get the predicted class index
        _, predicted_idx = torch.max(outputs, 1)
        
        # Map the index to the class name
        predicted_class_name = class_names[predicted_idx.item()]
        
        print(f"\nPrediction for '{image_path}':")
        print(f"--> Predicted Class: {predicted_class_name} 🏷️")

    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- 5. Run Prediction ---
# IMPORTANT: Replace this with the actual path to an image you want to test
test_image_path = "C:/Users/himan/OneDrive/Desktop/Lung Cancer/test_images_from_google/test2.png"
predict_image(test_image_path)