import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    logger.info(f"Loading data from directory: {data_dir}")

    data_transform = {'train': transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
                ]),
                    'val': transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.ToTensor()
                        ])}
    dataset = datasets.ImageFolder(data_dir, transform=data_transform['train'])

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Dataset split into {train_size} \
                training and {val_size} validation samples.")
    return train_dataset, val_dataset

def plot_training(train_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_model(data_dir, output_dir):
    logger.info("Starting model training...")

    stop = 0.001
    prev_loss = 1
    train_dataset, val_dataset = load_data(data_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, drop_last=True)

    prev_model = None
    model = LeafDiseaseModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses = []
    val_accuracies = []
    for epoch in range(25):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.long().to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                _, predicted = torch.max(preds, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                

        accuracy = correct / total
        val_accuracies.append(accuracy)
        logger.info(f"Validation Accuracy: {accuracy * 100:.2f}%")

        # if prev_loss - loss < stop or loss > prev_loss:
        #     logger.info(f"Stopping early at epoch \
        #                 {epoch + 1} due to minimal loss improvement.")
            # break

        prev_model = model
        prev_loss = loss
        
    plot_training(train_losses, val_accuracies)
    model_path = os.path.join(output_dir, 'leaf_disease_model.pth')
    torch.save(prev_model.state_dict(), model_path)
    logger.info(f"Model weights saved to {model_path}")
    return prev_model


if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]  # 'images/Apple/'
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Training started with data from {data_dir}")
        model_out = train_model(data_dir, output_dir)
        logger.info('Training completed successfully. \
                    The model weights are saved!')
    else:
        logger.error('No data directory provided. \
                     Please add the path to the image folder as an argument.')
        print('Usage: python script.py <path_to_image_folder>')
