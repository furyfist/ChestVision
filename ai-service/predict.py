import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image, ImageStat 

# model setup
print("Setting up for predictions......")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

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
model = models.resnet50() 
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

# Loading the Saved Model Weights 
model_path = "lung_cancer_classifier.pth"
# Use map_location to ensure it loads even if you don't have a GPU right now
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded from {model_path} and set to evaluation mode.")

# Image Transformation Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- NEW HELPER FUNCTION ---
def is_likely_ct_scan(img_pil):
    """
    Checks if an image is grayscale (CT scan) or colorful (Random object).
    Returns True if it looks like a CT scan.
    """
    # Convert to HSV (Hue, Saturation, Value)
    hsv_img = img_pil.convert('HSV')
    
    # Get the average saturation of the image (0 = gray, 255 = very colorful)
    mean_saturation = ImageStat.Stat(hsv_img).mean[1]
    
    # If average saturation is greater than 25, it definitely has color
    if mean_saturation > 25:
        return False
    return True

# Load and Predict a Single Image 
def predict_image(image_path):
    try:
        # Open the image file
        img = Image.open(image_path)
        
        # NEW: CHECK BEFORE PREDICTING 
        if not is_likely_ct_scan(img):
            print(f"\nPrediction for '{image_path}':")
            print(f"--> Result: Invalid Image (Not a CT Scan) ❌")
            return # Stop here, don't run the model
        
        # If it passes the check, convert to RGB for the model
        img = img.convert('RGB')
        
        # Apply the transformations
        img_t = transform(img)
        
        # Add batch dimension
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