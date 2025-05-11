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
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
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
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x

# Define the transformation to match training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model(device):
    # Initialize model
    model = CarClassifier(num_classes=196)
    
    # Load the trained weights
    model_path = os.path.join(os.path.dirname(__file__), 'car_classifier.pth')
    print(f"Loading model from: {model_path}")
    
    try:
        # Load state dict and check its contents
        state_dict = torch.load(model_path, map_location=device)
        print("State dict keys:", state_dict.keys())
        
        # Check a few parameter shapes
        print("\nSome parameter shapes:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}")
            
        model.load_state_dict(state_dict)
        print("\nModel weights loaded successfully")
        
        # Verify some layer weights after loading
        print("\nVerifying weights after loading:")
        for name, param in model.named_parameters():
            print(f"{name} - Mean: {param.mean().item():.4f}, Std: {param.std().item():.4f}")
            
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device, class_dict):
    try:
        # Verify model is in eval mode
        if model.training:
            model.eval()
        print(f"\nModel training mode: {model.training}")
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        print(f"Input image size: {image.size}")
        
        # Transform and verify tensor
        image_tensor = transform(image)
        print(f"Transformed tensor shape: {image_tensor.shape}")
        print(f"Tensor range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")
        
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get features
            features = model.feature_extractor(image_tensor)
            print(f"Feature shape: {features.shape}")
            print(f"Feature stats - Mean: {features.mean():.4f}, Std: {features.std():.4f}")
            
            # Flatten and get output
            features_flat = torch.flatten(features, 1)
            output = model.output(features_flat)
            print(f"Raw output shape: {output.shape}")
            print(f"Raw output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
            
            # Apply temperature scaling to soften the predictions
            temperature = 2.0
            scaled_output = output / temperature
            
            # Get probabilities with softmax
            probabilities = torch.nn.functional.softmax(scaled_output, dim=1)
            print(f"Probability sum: {probabilities.sum().item():.4f}")  # Should be close to 1.0
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            print("\nTop 3 predictions:")
            predictions = []
            for i in range(3):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                class_name = class_dict.get(str(idx), f"Unknown class {idx}")
                print(f"  {i+1}. {class_name}: {prob:.4f}")
                predictions.append((class_name, prob))
            
            # Return highest confidence prediction if above threshold
            if predictions[0][1] > 0.1:  # 10% confidence threshold
                return predictions[0][0]
            else:
                return "Confidence too low for reliable prediction"
            
    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        raise
