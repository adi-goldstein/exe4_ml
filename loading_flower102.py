import os
import random
import shutil
import glob
import tarfile
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models
from collections import defaultdict

base_dir = "/home/goldstad/exe4_ml/data/"

# Load dataset
dataset = datasets.Flowers102(root=base_dir, download=True)

# Define directory structure
splits = ["train", "val", "test"]

# Create base directories
for split in splits:
    for i in range(102):  # Flowers102 has 102 classes
        os.makedirs(os.path.join(base_dir, split, f"class_{i}"), exist_ok=True)

# Group images by class
class_images = defaultdict(list)
for i, (image, label) in enumerate(dataset):
    class_images[label].append(image)  # Store only the image, label is the key

# Ensure each class gets at least one image in each split
train_data, val_data, test_data = [], [], []

for label, images in class_images.items():
    num_images = len(images)

    if num_images < 3:
        # Distribute images evenly if there are fewer than 3
        split_sizes = [1] * num_images + [0] * (3 - num_images)
    else:
        # Apply 50%-25%-25% split
        train_size = max(1, int(0.5 * num_images))
        val_size = max(1, int(0.25 * num_images))
        test_size = num_images - train_size - val_size

    # Assign images while keeping track of their label
    train_data.extend([(img, label) for img in images[:train_size]])
    val_data.extend([(img, label) for img in images[train_size:train_size + val_size]])
    test_data.extend([(img, label) for img in images[train_size + val_size:]])

# Function to save images
def save_images(dataset, split_name):
    for i, (image, label) in enumerate(dataset):  # Properly unpack (image, label)
        class_dir = os.path.join(base_dir, split_name, f"class_{label}")
        image_path = os.path.join(class_dir, f"{i}.jpg")
        image.save(image_path)  # Now image is correctly assigned

# Save images to respective folders
save_images(train_data, "train")
save_images(val_data, "val")
save_images(test_data, "test")

if os.path.exists(base_dir+'/flowers-102'):
    shutil.rmtree(base_dir+'/flowers-102')
