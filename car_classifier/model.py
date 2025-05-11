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
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Print model architecture before loading weights
        print("\nModel architecture:")
        for name, module in model.named_children():
            print(f"{name}:")
            print(module)
        
        # Print some layer statistics before loading
        print("\nBefore loading weights:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}")
        
        # Load weights
        model.load_state_dict(state_dict)
        
        # Print layer statistics after loading
        print("\nAfter loading weights:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}")
                
        # Verify feature extractor weights
        first_conv = model.feature_extractor[0].weight
        print(f"\nFirst conv layer stats:")
        print(f"Shape: {first_conv.shape}")
        print(f"Mean: {first_conv.mean():.4f}")
        print(f"Std: {first_conv.std():.4f}")
        print(f"Min: {first_conv.min():.4f}")
        print(f"Max: {first_conv.max():.4f}")
        
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device, class_dict):
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        print(f"Input image size: {image.size}")
        
        # Transform and verify tensor
        image_tensor = transform(image)
        print(f"Transformed tensor shape: {image_tensor.shape}")
        print(f"Tensor range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")
        
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get features and output
            features = model.feature_extractor(image_tensor)
            features_flat = torch.flatten(features, 1)
            output = model.output(features_flat)
            
            # Print raw output stats
            print(f"Raw output shape: {output.shape}")
            print(f"Raw logits range: [{output.min().item():.2f}, {output.max().item():.2f}]")
            
            # Try different temperature scaling
            temperature = 0.5  # Lower temperature makes predictions more confident
            scaled_output = output / temperature
            
            # Get probabilities with log_softmax for numerical stability
            log_probs = torch.nn.functional.log_softmax(scaled_output, dim=1)
            probabilities = torch.exp(log_probs)
            
            print(f"Probability sum: {probabilities.sum().item():.4f}")
            
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            print("\nTop 5 predictions:")
            predictions = []
            for i in range(5):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                class_name = class_dict.get(str(idx), f"Unknown class {idx}")
                print(f"  {i+1}. {class_name}: {prob:.4f}")
                predictions.append((class_name, prob))
            
            # Return prediction with confidence threshold
            if predictions[0][1] > 0.05:  # Lower threshold to 5%
                return predictions[0][0]
            else:
                return "Confidence too low for reliable prediction"
            
    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        raise
