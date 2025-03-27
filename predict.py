#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# predict.py

# -------------------------------------------------------------------
# PROGRAMMER: C. Vasilov
# DATE CREATED: 22 March 2024
# REVISED DATE: 26 March 2025
#
# PURPOSE: Command-line script for image classification inference.
#
#          This script loads a trained flower classification model from a checkpoint,
#          processes an input image selected by folder number and image index, and prints
#          the top K predicted classes along with their probabilities. It also provides
#          a visualization showing the input image alongside a bar chart of predictions.
#
#          Users can specify hyperparameters such as folder number, image index, data
#          directory, top K predictions, and whether to use GPU for inference.
#
# USAGE: Run from the command line:
#        python predict.py final_best_model.pth --folder_number 6 --image_index 3 --top_k 5 --category_names cat_to_name.json --gpu
# -------------------------------------------------------------------

import os
import argparse
import torch
import json
from torchvision import datasets, transforms
from model_utils import load_checkpoint, predict, display_predictions

def get_predict_args():
    parser = argparse.ArgumentParser(description="Predict flower class from an image using a trained model.")
    parser.add_argument('checkpoint', type=str, help="Path to the model checkpoint.")
    
    # Instead of a fixed default, auto-detect the script’s folder:
    script_dir = os.path.dirname(os.path.abspath(__file__))  # The folder containing predict.py
    
    parser.add_argument('--folder_number', type=int, default=1,
                        help="Flower category folder number (1 to 102).")
    parser.add_argument('--image_index', type=int, default=0,
                        help="Index of the image within that folder.")
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(script_dir),
                        help="Root directory containing the 'flowers' folder. Defaults to the script's folder.")
    parser.add_argument('--top_k', type=int, default=5, help="Return top K predictions.")
    parser.add_argument('--category_names', type=str, default="cat_to_name.json",
                        help="Path to JSON file mapping class labels to flower names.")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference if available.")
    
    return parser.parse_args()

def main():
    args = get_predict_args()
    
    # Determine device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load the model checkpoint and move to device.
    model = load_checkpoint(args.checkpoint)
    model.to(device)
    
    # Build the test directory path.
    # We assume the test images are in: {data_dir}\flowers\test
    test_dir = os.path.join(args.data_dir, "flowers", "test")
    
    # Build a test dataset (needed for the class-to-index mapping).
    data_transform_validtest = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transform_validtest)
    
    # Run prediction using the folder number and image index.
    probs, classes = predict(folder_number=args.folder_number,
                             image_index=args.image_index,
                             subset_dir=test_dir,
                             dataset=test_dataset,
                             model=model,
                             topk=args.top_k)
    
    # Load category-to-name mapping.
    with open(args.category_names, "r") as f:
        cat_to_name = json.load(f)
    flower_names = [cat_to_name.get(cls, cls) for cls in classes]
    
    print("\n✅ Prediction Results:")
    print("Top K Probabilities: ", probs)
    print("Predicted Classes: ", classes)
    print("Flower Names: ", flower_names)
    
    # Optionally, display a visualization of the predictions.
    display_predictions(folder_number=args.folder_number,
                        image_index=args.image_index,
                        subset_dir=test_dir,
                        dataset=test_dataset,
                        model=model,
                        cat_to_name=cat_to_name,
                        topk=args.top_k)

if __name__ == "__main__":
    main()
