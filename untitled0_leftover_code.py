# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:48:51 2025

@author: camel
"""

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
with open('cat_to_name (1).json', 'r') as f:
    cat_to_name_1 = json.load(f)
    
cd /c/Users/camel/Desktop/Projects/MY_COURSE/second_project/udacity_AIPND_train_flower_network

# Force garbage collection to free memory
gc.collect()

# Get command-line arguments
args = get_input_args()

# Select model dynamically
if args.model == "resnet18":
    model = models.resnet18(pretrained=True)
    input_features = model.fc.in_features  # Get correct input size
    model.fc = nn.Linear(input_features, 102)  # Modify last layer for 102 classes

elif args.model == "resnet50":
    model = models.resnet50(pretrained=True)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 102)

elif args.model == "vgg16":
    model = models.vgg16(pretrained=True)
    input_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, 4096)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(4096, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

elif args.model == "vgg19":
    model = models.vgg19(pretrained=True)
    input_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, 4096)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(4096, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

elif args.model == "mobilenet":
    model = models.mobilenet_v2(pretrained=True)
    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(input_features, 102),
        nn.LogSoftmax(dim=1)
    )

# Move model to CPU or GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print model summary
print(f"\n✅ Loaded {args.model} successfully!\n")
print(model)