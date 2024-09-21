#!/usr/bin/env python3

import torch
from PIL import Image
from torchvision import transforms
from train import LeafDiseaseModel
import sys
import logging
import os
import json

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_path, device='cpu'):
    logger.info(f"Loading model from {model_path} on device {device}")
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} not found.")
        sys.exit(1)
    model = LeafDiseaseModel()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    return model


def preprocess_image(image_path, transform):
    logger.info(f"Preprocessing image {image_path}")
    if not os.path.exists(image_path):
        logger.error(f"Image file {image_path} not found.")
        sys.exit(1)

    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        logger.info("Image preprocessed successfully")
        return image
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        sys.exit(1)


def predict(image_path, model, device='cpu'):
    logger.info(f"Running prediction on image {image_path}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    image_tensor = preprocess_image(image_path, transform)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        try:
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            logger.info(f"Prediction complete: {predicted.item()}")
            return predicted.item()
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1:  # Image path passed as argument
        image_path = sys.argv[1]
        model_path = 'output/leaf_disease_model.pth'
        with open('map_classes.json', 'r') as jf:
            map_json = json.load(jf)
        logger.info(f"Starting inference on {image_path}")
        model = load_model(model_path, device)
        predicted_class = predict(image_path, model, device)
        print(f'Predicted class: {map_json[str(predicted_class)]}')
    else:
        logger.error('No image path provided as an argument.')
        print('Usage: python script.py <path_to_image>')
