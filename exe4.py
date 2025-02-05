import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import scipy
from tqdm import tqdm
import subprocess
import pandas as pd

# Preprocessing transformations
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder("data/train", transform=transform_train)
val_dataset = datasets.ImageFolder("data/val", transform=transform_val)
test_dataset = datasets.ImageFolder("data/test" ,transform = transform_val)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Number of classes (102 flower categories)
num_classes = 102

# ------------------------------------
# VGG19 Model
# ------------------------------------
def build_vgg19(device):

    #Load the pre-trained VGG19 model
    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    #Freeze all layers except the last few
    for param in vgg19.parameters():
        param.requires_grad = False
    #Only train the final classifier layer
    for param in vgg19.classifier[6].parameters():
        param.requires_grad = True

    #Dropout
    vgg19.classifier[6] = nn.Linear(4096, 102)
    vgg19.classifier[5] = nn.Dropout(p=0.5)

    vgg19 = vgg19.to(device)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg19.classifier.parameters(), lr=0.001, weight_decay=0.000001)

    return vgg19, criterion, optimizer

# ------------------------------------
# Training Function
# ------------------------------------
def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    history = {"train_loss": [], "val_loss": [],"train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):

        model.train()
        train_loss, correct_preds, total_preds = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100):
            
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            _, preds = outputs.max(1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                # Accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        train_loss /= len(train_loader.dataset)
        train_acc = correct_preds / total_preds
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    return history

# ------------------------------------
# Evaluate Function
# ------------------------------------
def evaluate_model(model,device, test_loader):
    model.eval()
    correct = 0
    total = 0
    probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities
            probabilities.append(probs)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return probabilities

def plot_vgg19_metrics(history, save_path=None):
    """
    Plots accuracy and loss graphs from the training history.

    Args:
        history (dict): A dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path (str): File path to save the plots (optional).
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_yolo_metrics(file,save_path):
    # Load metrics from CSV
    metrics = pd.read_csv(file,delimiter=",", skipinitialspace=True)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics['epoch'], metrics['train/loss'], label='Train Loss')
    plt.plot(metrics['epoch'], metrics['val/loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics['epoch'], metrics['train/accuracy'], label='Train Accuracy')
    plt.plot(metrics['epoch'], metrics['metrics/accuracy_top5'], label='Validation Accuracy top 5')
    plt.plot(metrics['epoch'], metrics['metrics/accuracy_top1'], label='Validation Accuracy top 1')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(save_path)

# ------------------------------------
# Main Workflow
# ------------------------------------

#### vgg19 ####

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg19_model, criterion, vgg19_optimizer = build_vgg19(device)

# Train VGG19
print("Training VGG19...")
vgg19_history = train_model(vgg19_model, device, train_loader, val_loader, criterion, vgg19_optimizer,num_epochs=10)

# Evaluate
print("Evaluating VGG19...")
vgg19_probs = evaluate_model(vgg19_model, device, test_loader)

plot_vgg19_metrics(vgg19_history, save_path="metrics_vgg19.png")

## yolov5 ###

subprocess.run(["python", "yolov5/classify/train.py", "--model", "yolov5s-cls.pt", "--data", "data", "--epochs", "10", "--img", "224", "--cache"])

subprocess.run(["python", "yolov5/classify/val.py", "--weights", "yolov5/runs/train-cls/exp/weights/best.pt", "--data", "data"])

plot_yolo_metrics("yolov5/runs/train-cls/exp/results.csv", "metrics_yolo.png")
