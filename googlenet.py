import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import googlenet, GoogLeNet_Weights
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np 
from mpl_toolkits.axes_grid1 import ImageGrid

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(r'C:\Users\ronjd\OneDrive\Desktop\comp_V\brain_tumor\Training', transform=data_transforms)
test_dataset = ImageFolder(r'C:\Users\ronjd\OneDrive\Desktop\comp_V\brain_tumor\Testing', transform=data_transforms)
size = len(train_dataset)
train_size = int(0.85 * size) 
val_size = size - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#Visualization
IMAGE_PER_CATEGORY = 4
CATEGORIES = train_dataset.classes
NUM_OF_CATEGORIES = len(CATEGORIES)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

category_images = {cat: [] for cat in CATEGORIES}
for images, labels in train_loader:
    for img, label in zip(images, labels):
        cat = CATEGORIES[label]
        if len(category_images[cat]) < IMAGE_PER_CATEGORY:
            category_images[cat].append(img)
    if all(len(imgs) >= IMAGE_PER_CATEGORY for imgs in category_images.values()):
        break

fig = plt.figure(figsize=(12, 8))
grid = ImageGrid(fig, 111, nrows_ncols=(IMAGE_PER_CATEGORY, NUM_OF_CATEGORIES), axes_pad=0.2)

i = 0
for col, category in enumerate(CATEGORIES):
    for row in range(IMAGE_PER_CATEGORY):
        ax = grid[i]
        img_tensor = category_images[category][row]
        img = img_tensor.numpy().transpose((1, 2, 0))
        img = (img * std + mean).clip(0, 1)
        ax.imshow(img)
        ax.axis('off')
        if row == IMAGE_PER_CATEGORY - 1:
            ax.set_title(category, fontsize=10, backgroundcolor='white')
        i += 1

plt.show()

num_classes = len(train_dataset.classes)
# Use GoogLeNet pretrained weights and preprocessing
weights = GoogLeNet_Weights.DEFAULT
model = googlenet(weights=weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


# Training setup
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, memory_format=torch.contiguous_format)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return 100 * correct / total

def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device, memory_format=torch.contiguous_format)
            labels = labels.to(device)
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
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

# Train and test
train_model(model, train_loader, val_loader, epochs=5)
test_acc = evaluate(model, test_loader)
print(f"\n\U0001F9EA Final Test Accuracy: {test_acc:.2f}%")