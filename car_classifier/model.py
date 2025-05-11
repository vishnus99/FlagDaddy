import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os

class CarClassifier(nn.Module):
    def __init__(self, num_classes=196):
        super(CarClassifier, self).__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=False)
        
        # Create feature extractor
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))  # Add adaptive pooling layer
        )
        
        # Create output layers
        self.output = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)  # flatten after pooling
        x = self.output(x)
        return x

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(device):
    # Initialize model with correct architecture
    model = CarClassifier(num_classes=196)
    
    # Load the trained weights
    model_path = os.path.join(os.path.dirname(__file__), 'car_classifier.pth')
    print(f"Loading model from: {model_path}")
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    model = model.to(device)
    model.eval()
    return model

# Create transform pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model, image_path, device, class_dict):
    try:
        # Verify model is in eval mode
        if model.training:
            model.eval()
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get predictions
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            # Load class dictionary
            dict_path = os.path.join(os.path.dirname(__file__), 'class_dict.json')
            with open(dict_path, 'r') as f:
                class_dict = json.load(f)
            
            print("\nTop 3 predictions:")
            for i in range(3):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                class_name = class_dict.get(str(idx), f"Unknown class {idx}")
                print(f"  {i+1}. {class_name}: {prob:.4f}")
            
            return class_dict.get(str(top_indices[0][0].item()), "Unknown")
            
    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        raise
