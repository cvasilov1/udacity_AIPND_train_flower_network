#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train.py

# -------------------------------------------------------------------
# PROGRAMMER: C. Vasilov
# DATE CREATED: 10 March 2025
# REVISED DATE: 26 March 2025
#
# PURPOSE: Command-line script to train a flower classification model.
#
#          This script loads the dataset (with training, validation, and test
#          subfolders), builds a pretrained CNN (either resnet50 or resnet18) with
#          a custom classifier, and trains the classifier using the provided
#          hyperparameters. It also evaluates the model on test data and saves the
#          best performing model checkpoint.
#
#          Users can configure hyperparameters such as learning rate, weight decay,
#          hidden units, dropout rate, epochs, batch size, scheduler parameters, and
#          whether to use the GPU.
#
# USAGE: Run from the command line:
# python train.py --gpu --arch resnet50 --hidden_units 512 --dropout 0.2 \ 
#                 --learning_rate 0.003 --weight_decay 5e-5 --epochs 30 --batch_size 32 --step_size 5 --gamma 0.1 
# -------------------------------------------------------------------

import os
from model_utils import (get_train_args, load_data, build_model, train_model,
                         evaluate_on_test, update_checkpoint)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def main():
    args = get_train_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Build the test directory path.
    # We assume the train images are in: {data_dir}\train
    train_dir = os.path.join(args.data_dir, "train")
    
    # Load data with the given batch size.
    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader, validtest_transform = load_data(args.data_dir, batch_size=args.batch_size)
    
    # Build model using specified architecture, hidden_units, and dropout.
    model = build_model(args.arch, args.hidden_units, args.dropout)
    model.to(device)
    print(f"\n✅ Using model architecture: {args.arch}")
    print(model)
    
    # Define loss function, optimizer (with weight decay), and scheduler.
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # Train the model (with epoch visualization).
    model = train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, args.epochs)
    
    # Evaluate on the test set.
    evaluate_on_test(model, test_dataset, test_loader)
    
    # Update and save the final checkpoint.
    update_checkpoint("best_model.pth", model, optimizer, args.epochs, train_dataset, args.save_dir, args.arch, args.hidden_units)
    
if __name__ == "__main__":
    main()