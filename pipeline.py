import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import googlenet, GoogLeNet_Weights

class ImagePipeline:
    def __init__(self, size, batch_size, device):
        self.size = size
        self.batch_size = batch_size
        self.device = device
    
    def load_preprocess_images(self, path):

        transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(path, transform=transform)
        num_classes = len(dataset.classes)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader, num_classes
    
    def evaluate(self, model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, memory_format=torch.contiguous_format)
                labels = labels.to(self.device)
                outputs = model(images)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
        return 100 * correct / total

    def train_model(self, model, train_loader, val_loader, criterion, optimizer, epochs=5):
        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                images = images.to(self.device, memory_format=torch.contiguous_format)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
            train_acc = 100 * correct / total
            val_acc = self.evaluate(model, val_loader)
            print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, "
                f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

def run_config(config):
    config = list(config)

    train_path = "dataset/Training/"
    test_path = "dataset/Testing/"

    device = torch.device("cpu")
    pipe = ImagePipeline((224, 224), config[0], device)
    train_loader, num_classes = pipe.load_preprocess_images(train_path)
    test_loader, _ = pipe.load_preprocess_images(test_path)

    size = len(train_loader.dataset)
    train_size = int(0.85 * size)
    val_size = size - train_size

    train_dataset, val_dataset = random_split(train_loader.dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config[0], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config[0], shuffle=False)

    if config[2] == "ResNet":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif config[2] == "GoogleNet":
        weights = GoogLeNet_Weights.DEFAULT
        model = googlenet(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif config[2] == "EfficientNet":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    pipe.train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5)
    test_acc = pipe.evaluate(model, test_loader)
    print(f"\n🧪 Final Test Accuracy: {test_acc:.2f}%")

config={
    16, # batch size
    0.85, # training size
    "EfficientNet" # model
}

run_config(config)


