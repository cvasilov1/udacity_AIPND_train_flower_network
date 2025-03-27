#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# model_utils.py

# -------------------------------------------------------------------
# PROGRAMMER: C. Vasilov
# DATE CREATED: 28 February 2024
# REVISED DATE: 26 March 2025
#
# PURPOSE: This module contains common functions and utilities for both
#          training and inference of a flower classification model.
#
#          It includes functions for argument parsing, data loading and
#          augmentation (with transforms), model building (with customizable
#          hyperparameters), training and evaluation (with logging and early
#          stopping), checkpoint saving/loading, image processing, prediction,
#          and visualization of results.
#
# USAGE: Import the functions from this module into train.py and predict.py.
# -------------------------------------------------------------------

import os
import sys
import json
import argparse
import time
import copy
import gc
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------
# Argument Parsing Functions
# ------------------------

import argparse
import os

def get_train_args():
    """
    Parses command-line arguments for training the image classifier.
    Returns:
      args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a flower classifier.")

    # The directory containing train.py (or model_utils.py).
    default_dir = os.path.dirname(os.path.abspath(__file__))

    # Build a default path that points to "flowers" inside the script's directory.
    default_data_dir = os.path.join(default_dir, "flowers")

    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                        help="Path to the dataset root directory (with 'train', 'valid', 'test'). "
                             "Defaults to <this_script_directory>/flowers.")

    parser.add_argument('--save_dir', type=str, default=default_dir,
                        help="Directory to save checkpoints. Defaults to the script's directory.")

    parser.add_argument('--arch', type=str, default="resnet50", choices=["resnet18", "resnet50"],
                        help="Pretrained model architecture to use.")
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help="Learning rate for training.")
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help="Weight decay (L2 regularization) for optimizer.")
    parser.add_argument('--hidden_units', type=int, default=512,
                        help="Number of hidden units in the classifier.")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="Dropout probability for classifier.")
    parser.add_argument('--epochs', type=int, default=30,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for DataLoader.")

    # GPU/CPU: Use GPU by default, allow override.
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu', dest='gpu', action='store_true',
                       help="Use GPU for training (default).")
    group.add_argument('--no-gpu', dest='gpu', action='store_false',
                       help="Force CPU for training.")
    parser.set_defaults(gpu=True)

    # Scheduler parameters:
    parser.add_argument('--step_size', type=int, default=5,
                        help="Step size for learning rate scheduler.")
    parser.add_argument('--gamma', type=float, default=0.1,
                        help="Gamma for learning rate scheduler.")

    args = parser.parse_args()
    return args

def get_predict_args():
    parser = argparse.ArgumentParser(description="Predict flower class using a trained model.")
    parser.add_argument('checkpoint', type=str, help="Path to the model checkpoint (e.g., final_best_model.pth).")
    
    # For selecting an image by folder and index:
    parser.add_argument('--folder_number', type=int, default=5,
                        help="Flower category folder number (1 to 102) in the test set. Default is 5.")
    parser.add_argument('--image_index', type=int, default=2,
                        help="Index of the image within the folder (0 for first image). Default is 2.")
    
    # Smart default for data_dir: use the directory where predict.py lives and append 'flowers'
    default_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(default_dir, "flowers")
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                        help="Root directory of the dataset. Defaults to <this_script_directory>/flowers.")
    
    parser.add_argument('--top_k', type=int, default=5, help="Return top K predictions.")
    parser.add_argument('--category_names', type=str, default="cat_to_name.json",
                        help="Path to JSON file mapping class labels to flower names.")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu', dest='gpu', action='store_true', help="Use GPU for inference (default).")
    group.add_argument('--no-gpu', dest='gpu', action='store_false', help="Force CPU for inference.")
    parser.set_defaults(gpu=True)
    
    return parser.parse_args()

# ------------------------
# Data Loading and Transforms
# ------------------------

def load_data(data_dir, batch_size=32):
    """
    Loads the training, validation, and test datasets along with DataLoaders.
    
    Returns:
      train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader,
      data_transform_validtest: the transform used for validation/test.
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir  = os.path.join(data_dir, 'test')
    
    data_transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    data_transform_validtest = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transform_train)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transform_validtest)
    test_dataset  = datasets.ImageFolder(test_dir, transform=data_transform_validtest)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader, data_transform_validtest

# ------------------------
# Model Building and Training Functions
# ------------------------

def build_model(arch, hidden_units, dropout):
    """
    Loads a pretrained model and rebuilds its classifier.
    
    Returns:
      model: The model with a new classifier.
    """
    if arch == "resnet18":
        model = models.resnet18(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    input_features = model.fc.in_features
    model.fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(p=dropout)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    return model

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, epochs, patience=3):
    """
    Trains the model with early stopping based on validation loss.
    Prints training and validation loss/accuracy for each epoch.
    
    Returns:
      model: The trained model.
    """
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(output, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct / total * 100
        
        # Validation phase.
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                val_loss += loss.item()
                _, preds = torch.max(output, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / len(valid_loader)
        val_acc = correct / total * 100
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Model improved! Best model saved.")
        else:
            patience_counter += 1
            print(f"🚨 No improvement for {patience_counter} epoch(s).")
        
        if patience_counter >= patience:
            print("\n⏹️ Early stopping triggered.")
            break
        
        scheduler.step()
    
    return model

def evaluate_on_test(model, test_dataset, test_loader):
    """
    Evaluates the model on the test set and prints test accuracy.
    
    Returns:
      test_accuracy: The overall test accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to("cpu"), labels.to("cpu")
            output = model(images)
            probs = torch.exp(output)
            _, preds = probs.topk(1, dim=1)
            predicted_classes = [idx_to_class[idx.item()] for idx in preds.squeeze()]
            actual_classes = [idx_to_class[idx.item()] for idx in labels]
            correct += sum(p == a for p, a in zip(predicted_classes, actual_classes))
            total += labels.size(0)
    test_acc = (correct / total) * 100
    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
    return test_acc

def update_checkpoint(filepath, model, optimizer, epochs, dataset, save_dir, arch, hidden_units):
    """
    Loads a model checkpoint, adds extra info (optimizer state, hyperparameters, class_to_idx),
    and saves the updated checkpoint.
    """
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    new_checkpoint = {
        "arch": arch,
        "state_dict": checkpoint,
        "optimizer_state": optimizer.state_dict(),
        "epochs": epochs,
        "class_to_idx": dataset.class_to_idx,
        "hyperparameters": {
            "learning_rate": optimizer.param_groups[0]["lr"],
            "dropout": new_dropout,
            "hidden_units": hidden_units
        }
    }
    new_filepath = os.path.join(save_dir, "final_best_model.pth")
    torch.save(new_checkpoint, new_filepath)
    print(f"\n✅ Checkpoint updated and saved as: {new_filepath}")


def load_checkpoint(filepath):
    """
    Loads a trained model from a checkpoint file and rebuilds the model.
    
    Returns:
      model: The rebuilt model in evaluation mode.
    """
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    arch = checkpoint.get("arch", "resnet50")
    if arch == "resnet18":
        model = models.resnet18(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    input_features = model.fc.in_features
    hidden_units = checkpoint.get("hyperparameters", {}).get("hidden_units", 512)
    model.fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.class_to_idx = checkpoint["class_to_idx"]
    model.eval()
    return model

# ------------------------
# Image Processing and Visualization Functions
# ------------------------

def process_image(folder_number, image_index, subset_dir, transform):
    """
    Loads an image from a specified folder and image index, applies the given transform,
    and returns:
      - original_image (PIL.Image.Image): The original image (before transformation).
      - processed_image (torch.Tensor): The image tensor after applying the transform.
      - denormalized_image (numpy.ndarray): The de-normalized image (H, W, C) for visualization.
      
    Parameters:
      folder_number (int): Flower category folder (1 to 102). Defaults to 1 if out of range.
      image_index (int): Index of the image in the folder (defaults to 0 if out of range).
      subset_dir (str): Directory of the dataset subset (e.g., train, valid, or test).
      transform (callable): Transform to apply (should include normalization).
    """
    if folder_number < 1 or folder_number > 102:
        print(f"Warning: folder_number {folder_number} is out of range. Defaulting to 1.")
        folder_number = 1
    class_folder = os.path.join(subset_dir, str(folder_number))
    if not os.path.isdir(class_folder):
        raise ValueError(f"Folder '{class_folder}' does not exist!")
    image_files = sorted(os.listdir(class_folder))
    if image_index >= len(image_files):
        print(f"Warning: image_index {image_index} is out of range for folder '{class_folder}'. Defaulting to 0.")
        image_index = 0
    image_path = os.path.join(class_folder, image_files[image_index])
    original_image = Image.open(image_path)
    processed_image = transform(original_image)
    denorm_image = processed_image.clone().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    denormalized_image = denorm_image * std + mean
    denormalized_image = np.clip(denormalized_image, 0, 1)
    return original_image, processed_image, denormalized_image

def imshow(image, ax=None, title=None):
    """
    Displays a PyTorch tensor as an image.
    """
    if ax is None:
        fig, ax = plt.subplots()
    image = image.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    if title:
        ax.set_title(title)
    return ax

def predict(folder_number, image_index, subset_dir, dataset, model, topk=5, transform=None):
    """
    Predicts the top-K classes for an image using a trained model.
    
    Parameters:
      folder_number (int): Flower category folder (1 to 102)
      image_index (int): Index of the image in the folder
      subset_dir (str): Directory of the dataset subset (e.g., test)
      dataset: ImageFolder dataset (for class-to-index mapping)
      model: Trained PyTorch model (must have class_to_idx attribute)
      topk (int): Number of top predictions to return
      transform (callable): Optional; transform to apply. If None, uses a default validation transform.
      
    Returns:
      top_probs (list): Top-K probabilities.
      top_classnames (list): Corresponding class labels.
    """
    from torchvision import transforms
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    _, processed_image, _ = process_image(folder_number, image_index, subset_dir, transform)
    image_tensor = processed_image.unsqueeze(0).to(torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        output = model.forward(image_tensor)
    probs = torch.exp(output)
    top_probs, top_indices = probs.topk(topk, dim=1)
    top_probs = top_probs.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()
    idx_to_class = {value: key for key, value in dataset.class_to_idx.items()}
    top_classnames = [idx_to_class[idx] for idx in top_indices]
    return top_probs, top_classnames

def display_predictions(folder_number, image_index, subset_dir, dataset, model, cat_to_name, topk=5, transform=None):
    """
    Displays an image alongside a horizontal bar chart of its top-K predicted classes.
    
    Parameters:
      folder_number (int): Flower category folder (1 to 102). Defaults to 1 if out of range.
      image_index (int): Index of the image in the folder. Defaults to 0 if out of range.
      subset_dir (str): Directory of the dataset subset (e.g., test)
      dataset: ImageFolder dataset (for class-to-index mapping)
      model: Trained PyTorch model
      cat_to_name (dict): Mapping from class labels to flower names
      topk (int): Number of top predictions to display
      transform (callable): Optional; transform to apply. If None, uses a default validation transform.
    """
    from torchvision import transforms
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    
    # Apply default logic for folder_number and image_index
    if folder_number < 1 or folder_number > 102:
        print(f"Warning: folder_number {folder_number} is out of range. Defaulting to 1.")
        folder_number = 1
    import os
    class_folder = os.path.join(subset_dir, str(folder_number))
    if not os.path.isdir(class_folder):
        raise ValueError(f"Folder '{class_folder}' does not exist!")
    image_files = sorted(os.listdir(class_folder))
    if image_index >= len(image_files):
        print(f"Warning: image_index {image_index} is out of range for folder '{class_folder}'. Defaulting to 0.")
        image_index = 0
    
    original_image, processed_image, denorm_image = process_image(folder_number, image_index, subset_dir, transform)
    actual_class = str(folder_number)
    actual_label = cat_to_name.get(actual_class, "Unknown")
    probs, classes = predict(folder_number, image_index, subset_dir, dataset, model, topk, transform)
    predicted_flowers = [cat_to_name.get(str(cls), str(cls)) for cls in classes]
    import numpy as np
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = [probs[i] for i in sorted_indices]
    sorted_flowers = [predicted_flowers[i] for i in sorted_indices]
    top_prediction = sorted_flowers[0]
    correct_prediction = (top_prediction == actual_label)
    title_color = "green" if correct_prediction else "red"
    correctness_marker = "- Right!" if correct_prediction else "- Wrong!"
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(denorm_image)
    axes[0].axis("off")
    axes[0].set_title(f"Actual: {actual_label} {correctness_marker}", fontsize=14, fontweight="bold", color=title_color)
    axes[1].barh(sorted_flowers[::-1], sorted_probs[::-1], color="skyblue")
    axes[1].set_xlabel("Prediction Probability", fontsize=12)
    axes[1].set_xlim(0, 1)
    axes[1].set_title(f"Top-{topk} Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
