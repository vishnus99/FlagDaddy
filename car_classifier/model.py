import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import logging
import os
import traceback
import torch.nn.functional as F
import json

# Model definition
class CarClassifier(torch.nn.Module):
    def __init__(self, num_classes: int, train_resnet: bool = False):
        super().__init__()
        model = models.resnet50(pretrained=True)
        
        self.feature_extractor = torch.nn.Sequential(
            *list(model.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if not train_resnet:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.output = nn.Sequential(
            nn.Linear(2048,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x).view(x.shape[0], -1)
        return self.output(features)

# Create transform pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model, image_path, device, class_dict):
    try:
        # 1. Verify model state
        print(f"\nModel device: {next(model.parameters()).device}")
        print(f"Input device: {device}")
        print(f"Model training mode: {model.training}")
        
        # 2. Load and verify image
        image = Image.open(image_path).convert('RGB')
        print(f"\nImage size: {image.size}")
        print(f"Image mode: {image.mode}")
        
        # 3. Check transformation
        image_tensor = transform(image)
        print(f"\nTransformed tensor shape: {image_tensor.shape}")
        print(f"Tensor value range: ({image_tensor.min():.2f}, {image_tensor.max():.2f})")
        
        # 4. Prepare batch
        image_tensor = image_tensor.unsqueeze(0).to(device)
        print(f"Input batch shape: {image_tensor.shape}")
        
        # 5. Get predictions
        model.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            output = model(image_tensor)
            print(f"\nRaw output shape: {output.shape}")
            print(f"Output value range: ({output.min():.2f}, {output.max():.2f})")
            
            # Get probabilities
            probabilities = F.softmax(output, dim=1)
            print(f"Probability sum: {probabilities.sum().item():.2f}")  # Should be close to 1.0
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            
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
