import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import logging
import os
import traceback

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
        # Load image
        logger.info("Loading image...")
        image = Image.open(image_path).convert('RGB')
        logger.info("Image loaded successfully")

        # Transform image
        logger.info("Transforming image...")
        image_tensor = transform(image).unsqueeze(0).to(device)
        logger.info("Image transformed successfully")

        # Make prediction
        logger.info("Running model prediction...")
        model.eval()
        with torch.no_grad():
            try:
                output = model(image_tensor)
                _, predicted = torch.max(output.data, 1)
                result = predicted.item()
                logger.info(f"Prediction successful: {result}")
                return result
            except Exception as e:
                logger.error(f"Model prediction failed: {str(e)}")
                logger.error(traceback.format_exc())
                raise ValueError(f"Model prediction failed: {str(e)}")

    except Exception as e:
        logger.error(f"Error in predict_image: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Error in predict_image: {str(e)}")
