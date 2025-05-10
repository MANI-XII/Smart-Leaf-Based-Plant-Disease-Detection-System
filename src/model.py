import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import os

class PlantDiseaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(weights='IMAGENET1K_V1')  # Updated weight parameter
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)

    def forward(self, xb):
        return self.network(xb)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Ensure consistent resizing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per ResNet training
])

# Class names
num_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Load the model
model = PlantDiseaseModel()
model_path = os.path.join(os.getcwd(), 'Models', 'plantDisease-resnet34.pth')

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model loaded successfully!")
else:
    print("❌ Model file not found! Please check the path:", model_path)

def predict_image(img_bytes):
    """
    Predict the disease class of a plant leaf image.

    Args:
        img_bytes (bytes): The image in byte format.

    Returns:
        str: Predicted plant disease class.
    """
    try:
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # Ensure image is in RGB format
        tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():  # Disable gradient calculations for inference
            yb = model(tensor)
            _, preds = torch.max(yb, dim=1)
        return num_classes[preds[0].item()]
    except Exception as e:
        return f"Error processing image: {str(e)}"
