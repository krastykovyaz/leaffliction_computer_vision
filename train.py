from PIL import Image, ImageEnhance, ImageOps
import random
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from zipfile import ZipFile
import shutil

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LeafDiseaseModel(nn.Module):
    def __init__(self):
        super(LeafDiseaseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data(data_dir):
    dataset = datasets.ImageFolder(data_dir, transform=data_transform['train'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def train_model(data_dir, output_dir):
    stop = 0.001
    prev_loss = 1
    train_dataset, val_dataset = load_data(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, drop_last=True)
    prev_model = None
    model = LeafDiseaseModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.float()
            batch_y = batch_y.long()
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                _, predicted = torch.max(preds, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        if prev_loss - loss < stop or loss > prev_loss:
            break
        prev_model = model
        prev_loss = loss
    model_path = os.path.join(output_dir, 'leaf_disease_model.pth')
    torch.save(prev_model.state_dict(), model_path)
    return prev_model


if __name__=='__main__':
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]# 'images/Apple/'
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        data_transform = {'train': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ]), 'val': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
            ])}
        model_out = train_model(data_dir, output_dir)  
        print('The weights of the model are saved!')
    else:
        print('Add path to the image foldes as an argument')
