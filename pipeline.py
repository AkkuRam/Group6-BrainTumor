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
import timm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchcam.methods import GradCAM
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask
from PIL import Image

class ImagePipeline:
    def __init__(self, size, batch_size, device):
        """
        :param size: the size to resize image (224, 224)
        :param batch_size: batch size was 16
        :param device: this uses the CPU
        """
        self.size = size
        self.batch_size = batch_size
        self.device = device
    
    def load_preprocess_images(self, path):
        """
        :param path: respective folder path for training / test set
        :returns: a dataloader with preprocessing applied to the dataset and number of 
        target classes in the dataset
        """
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
    
    def get_loaders(self, config):
        """
        :param config: represents parameters to choose which model to run, but here only batch size
        is used
        :returns: returns all loaders (train, val, test) and number of target classes
        """
        train_path = "dataset/Training/"
        test_path = "dataset/Testing/"
        train_loader, num_classes = self.load_preprocess_images(train_path)
        
        size = len(train_loader.dataset)
        train_size = int(0.85 * size)
        val_size = size - train_size

        train_dataset, val_dataset = random_split(train_loader.dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        test_loader, _ = self.load_preprocess_images(test_path)

        return train_loader, val_loader, test_loader, num_classes
    
    def evaluate(self, model, loader):
        """
        :param model: the chosen model (ResNet/EfficientNet/GoogleNet)
        :param loader: this is the test loader, since after training it is evaluated
        on the test set
        :returns: outputs accuracy on test set
        """
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
        """
        :param model: the chosen model (ResNet/EfficientNet/GoogleNet)
        :param train_loader: loader for the training set
        :param val_loader: loader for the validation set
        :param criterion: this is the loss function, which was cross entropy
        :param optimizer: the adam optimizer was used
        :param epochs: number of iterations to run evaluation on validation set
        :returns: returns the training and validation accuracy per epoch
        """
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

    def run_config(self, config, device, num_classes, train_loader, val_loader, test_loader):
        """
        :param config: the parameters for which model to run
        :param device: this uses the CPU
        :param num_classes: number of target classes
        :param train_loader: loader for training set
        :param val_loader: loader for validation set
        :param test_loader: loader for test_set
        :returns: this runs the config for training model on validation set and evaluating on test set, then 
        saves the models weights into "models_saved" folder 
        """
        
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        if config["model"] == "GoogleNet":
            weights = GoogLeNet_Weights.DEFAULT
            model = googlenet(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif config["model"] == "EfficientNet":
            weights = EfficientNet_B0_Weights.DEFAULT
            model = efficientnet_b0(weights=weights)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        self.train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5)
        torch.save(model.state_dict(), config["model_path"])
        test_acc = self.evaluate(model, test_loader)
        print(f"\n🧪 Final Test Accuracy: {test_acc:.2f}%")

def get_image_metrics(path, config, num_classes, class_names, device, test_loader):
    """
    :param path: path for the saved model weights
    :param config: the parameters for which model to run
    :param num_classes: number of target classes
    :param class_names: respective names for the target classes
    :param device: this uses the CPU
    :param test_loader: loader for the test set
    :returns: 
        - saves the confusion matrices of all models in "confusion_matrix" folder
        - correct & misclassified predictions for all models in "misclassified" folder
        - for the same images in "misclassified" folder it also saves feature importance
        plots from Grad-CAM in "feature_importance" folder
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if config["model"] == "GoogleNet":
        model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif config["model"] == "EfficientNet":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()


    all_preds = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(images.cpu())

    cm = confusion_matrix(all_labels, all_preds)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title("Confusion Matrix")
    plt.savefig(f'confusion_matrix/{config["model"]}.png', bbox_inches='tight')
    plt.close() 
    plt.tight_layout()  

    selected_correct, selected_incorrect = misclassified_images(config, all_images, all_preds, all_labels) 

    print("✅ Grad-CAM on Correct Predictions")
    for idx in selected_correct:
        get_feature_importance(model, config, all_images[idx], all_labels[idx], all_preds[idx], idx)

    print("❌ Grad-CAM on Incorrect Predictions")
    for idx in selected_incorrect:
        get_feature_importance(model, config, all_images[idx], all_labels[idx], all_preds[idx], idx)

def get_feature_importance(model, config, all_images, all_preds, all_labels, idx):
    """
    :param model: the chosen model (ResNet/EfficientNet/GoogleNet)
    :param config: the parameters for which model to run
    :param all_images: all test images
    :param all_preds: all predictions for these test images
    :param all_labels: all labels for those test images
    :param idx: the index from the 10 misclassified images, used here
    to view feature importance on them
    :returns: feature importance plots of those 10 images are saved in
    "feature_importance" folder
    """
    if config["model"] == "ResNet":
        blocks = list(model.layer4.children())
    elif config["model"] == "GoogleNet":
         blocks = model.inception5b
    elif config["model"] == "EfficientNet":
        blocks = list(model.features.children())[-1]
   
    cam_extractor = GradCAM(model, target_layer=blocks)

    image = all_images.unsqueeze(0).to(device)  
    output = model(image)
    class_idx = output.argmax().item()


    cam = cam_extractor(class_idx, output)[0].cpu()
    raw_img = all_images.cpu()
    to_img = to_pil_image(raw_img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                                    + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1))

    cam_img = overlay_mask(to_img, to_pil_image(cam, mode='F'), alpha=0.5)

    plt.imshow(cam_img)
    plt.title(f"True: {class_names[all_labels]} | Pred: {class_names[all_preds]}")
    plt.axis("off")
    plt.savefig(f"feature_importance/{config["model"]}{idx}.png")
    plt.close()
    plt.show()

    
def misclassified_images(config, all_images, all_preds, all_labels):
    """
    :param config: the parameters for which model to run
    :param all_images: all test images
    :param all_preds: all predictions for these test images
    :param all_labels: all labels for those test images
    :returns: saves 10 correctly and misclassified images in the 
    "misclassified" folder
    """
    correct_idx = [i for i in range(len(all_labels)) if all_labels[i] == all_preds[i]]
    incorrect_idx = [i for i in range(len(all_labels)) if all_labels[i] != all_preds[i]]

    selected_correct = correct_idx[:10]
    selected_incorrect = incorrect_idx[:10]

    def imshow(img, label, pred, is_wrong=False):
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
        plt.imshow(img)
        title = f"Pred: {class_names[pred]}\nTrue: {class_names[label]}"
        if is_wrong:
            plt.title(title, color='red')
        else:
            plt.title(title, color='green')
        plt.axis('off')


    fig_correct, axes_correct = plt.subplots(2, 5, figsize=(15, 6))
    fig_correct.suptitle("✅ Correct Predictions", fontsize=16)
    for ax, idx in zip(axes_correct.flatten(), selected_correct):
        plt.sca(ax)
        imshow(all_images[idx], all_labels[idx], all_preds[idx], is_wrong=False)
    plt.savefig(f'misclassified/correct_predictions_{config["model"]}.png')
    plt.close() 
    plt.tight_layout()


    fig_wrong, axes_wrong = plt.subplots(2, 5, figsize=(15, 6))
    fig_wrong.suptitle("❌ Incorrect Predictions", fontsize=16)
    for ax, idx in zip(axes_wrong.flatten(), selected_incorrect):
        plt.sca(ax)
        imshow(all_images[idx], all_labels[idx], all_preds[idx], is_wrong=True)
    plt.savefig(f'misclassified/incorrect_predictions_{config["model"]}.png')
    plt.close() 
    plt.tight_layout()

    return selected_correct, selected_incorrect


def save_model_weights(config, device, num_classes, train_loader, val_loader, test_loader):
    """
    :param config: the parameters for which model to run
    :param device: this uses the CPU
    :param num_classes: number of target classes
    :param train_loader: loader for training set
    :param val_loader: loader for validation set
    :param test_loader: loader for test_set
    :returns: saves the model weights in "models_saved" folder
    """
    pipe.run_config(config, device, num_classes, train_loader, val_loader, test_loader)

##############################################################################################
#                                        Testing                                             #
##############################################################################################

config_resnet={
    "batch_size": 16,
    "model": "ResNet",
    "model_path": "models_saved/resnet.pth" 
}

config_googlenet={
    "batch_size": 16, 
    "model": "GoogleNet", 
    "model_path": "models_saved/googlenet.pth" 
}

config_efficientnet={ 
    "batch_size": 16, 
    "model": "EfficientNet",
    "model_path": "models_saved/efficientnet.pth" 
}

device = torch.device("cpu")
pipe = ImagePipeline((224, 224), config_resnet["batch_size"], device)
train_loader, val_loader, test_loader, num_classes = pipe.get_loaders(config_resnet)
class_names = test_loader.dataset.classes
num_classes = len(class_names)

# you can uncomment these 3 commented lines below, if you want to run the pipeline and save the weights
# this is very slow around 1hr for each model - 3hrs in total
# save_model_weights(config_resnet, device, num_classes, train_loader, val_loader, test_loader)
# save_model_weights(config_googlenet, device, num_classes, train_loader, val_loader, test_loader)
# save_model_weights(config_efficientnet, device, num_classes, train_loader, val_loader, test_loader)

# these 3 lines are for getting all the respective images 
# i.e. feature importance plots, misclassified vs correctly classfied, confusion matrices
# this can take some time 5-10 mins
get_image_metrics("models_saved/resnet.pth", config_resnet, num_classes, class_names, device, test_loader)
get_image_metrics("models_saved/googlenet.pth", config_googlenet, num_classes, class_names, device, test_loader)
get_image_metrics("models_saved/efficientnet.pth", config_efficientnet, num_classes, class_names, device, test_loader)





