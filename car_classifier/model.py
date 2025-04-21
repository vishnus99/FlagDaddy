import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import logging
import os
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
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    force=True
    )
    logger = logging.getLogger('car_classifier_bot')
    logger.info(f"Predict image called with path: {image_path}")
    
    try:
        # Type checking
        if not isinstance(image_path, str):
            raise TypeError(f"Expected string path, got {type(image_path)}")
        
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model)}")
            
        # Verify file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Load and verify image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to open image: {str(e)}")
            
        # Transform image
        try:
            image_tensor = transform(image).unsqueeze(0).to(device)
        except Exception as e:
            raise ValueError(f"Failed to transform image: {str(e)}")
            
        model.eval()
        with torch.no_grad():
            try:
                image_tensor = transform(image).unsqueeze(0).to(device)
                output = model(image_tensor)
                _, predicted = torch.max(output.data, 1)
                return class_dict[predicted.item()]
            except Exception as e:
                raise ValueError(f"Failed to make prediction: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error in predict_image: {str(e)}")
