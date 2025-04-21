import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import logging
import os
import traceback
import json
import torch.nn.functional as F

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

def predict_image(model, image_path, device):
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    force=True
    )
    logger = logging.getLogger('car_classifier_bot')
    logger.info(f"Predict image called with path: {image_path}")
    
    try:
        # Load class dictionary
        with open('class_dict.json', 'r') as f:
            class_dict = json.load(f)
        print(f"Number of classes in dictionary: {len(class_dict)}")
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            # Get predictions
            output = model(image_tensor)
            print(f"Model output shape: {output.shape}")  # Should be [1, 196]
            
            # Get probabilities
            probabilities = F.softmax(output, dim=1)
            
            # Get top 3 predictions for debugging
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            print("\nTop 3 predictions:")
            for i in range(3):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                class_name = class_dict.get(str(idx), f"Unknown class {idx}")
                print(f"  {i+1}. {class_name}: {prob:.4f}")
            
            # Get the top prediction
            pred_idx = top_indices[0][0].item()
            class_name = class_dict.get(str(pred_idx), f"Unknown class {pred_idx}")
            
            return class_name
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
