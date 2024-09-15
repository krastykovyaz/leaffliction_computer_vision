import torch
from PIL import Image
from torchvision import datasets, transforms
from train import LeafDiseaseModel
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path, device='cpu'):
    model = LeafDiseaseModel()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def predict(image_path, model, device='cpu'):
    transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    image_tensor = preprocess_image(image_path, transform)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()
        

if __name__=='__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]# 'images/Apple/apple_black_rot/image (100).JPG'
        model_path = 'output/leaf_disease_model.pth'
        model = load_model(model_path)
        predicted_class = predict(image_path, model, device)
        print(f'Predicted class: {predicted_class}')
    else:
        print('Add path to the image file')