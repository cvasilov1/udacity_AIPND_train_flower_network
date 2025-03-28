# Flower Classifier Project

## Project Overview

This project builds a deep learning classifier that recognizes different species of flowers. The classifier uses a pre-trained Convolutional Neural Network (CNN) (either ResNet18 or ResNet50) with a custom feedforward classifier to predict one of 102 flower categories. The project demonstrates:

\- Data augmentation and normalization using torchvision transforms.

\- Building and training a model with frozen pre-trained layers.

\- Evaluation on a test set and checkpointing the model.

\- Inference via command-line scripts that select an image by folder number and image index.

The project supports both interactive development through a Jupyter Notebook and reproducible training and inference using command-line scripts.

## Project Structure

udacity_AIPND_train_flower_network/

- **Image Classifier Project.ipynb** # Notebook for interactive exploration
- **train.py** # CLI script for training the model
- **predict.py** # CLI script for making predictions
- **model_utils.py** # Module with common functions (data loading, model building, training, prediction, etc.)
- **cat_to_name.json** # JSON mapping from class labels to flower names
- **final_best_model.pth** # Checkpoint updated with hyperparameters and class-to-index mapping
- **LICENSE** # Project license
- **README.md** # This documentation
- **flowers** # Folder with data - not added to this repo (>300 MB), you must download the data and create it! See below.

## 🏵️ Dataset

This project uses the [Oxford 102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).

Due to size constraints, the dataset is **not included in this repository**.

To use the model:
1. Download the dataset from https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz (**Direct download link!**)
2. Unzip and place the folder in the root directory of this repo as `flowers/`.
3. Check that `flowers/` has sub-folders `train/`, `valid/`, `test/`.

## Using the Jupyter notebook

The Jupyter Notebook (Image Classifier Project.ipynb) provides an interactive environment to:

- Load and preprocess the flower dataset.
- Build, train, and evaluate the classifier step by step.
- Visualize data transformations and model predictions.

To run the notebook:

1. Open a terminal (with your environment activated) and navigate to the project folder.
2. Launch Jupyter Notebook: jupyter notebook. Alternatively, open Jupyter from Anaconda.
3. Open Image Classifier Project.ipynb and run the cells interactively.

## Using the command-line application

The project includes three Python scripts:

- **model_utils.py:** This module contains common functions and utilities for both training and inference of a flower classification model. It includes functions for argument parsing, data loading and augmentation (with transforms), model building (with customizable hyperparameters), training and evaluation (with logging and early stopping), checkpoint saving/loading, image processing, prediction, and visualization of results. This script will not be run by itself from the command line, but the functions defined in it will be imported into train.py and predict.py.
- **train.py:** Command line script for training the flower classifier.
- **predict.py:** Command line script for performing inference on a selected image.

### 🧠 Running train.py

This script loads the dataset (with training, validation, and test subfolders), builds a pretrained CNN (either resnet50 or resnet18) with a custom classifier, and trains the classifier using the provided hyperparameters. It also evaluates the model on test data and saves the best performing model checkpoint. Users can configure hyperparameters such as:
Please be careful when changing any of the hyperparameters, as it might negatively affect the training of the network!

- **Data Directory:** Defaults to the directory where train.py resides, with a subfolder flowers (which contains train, valid, and test).
- **Save Directory:** Defaults to the directory where train.py is located.
- **Model Architecture:** --arch (choices: resnet50 (default) or resnet18)
- **Learning Rate:** --learning_rate (default: 0.003)
- **Weight Decay:** --weight_decay (default: 5e-5)
- **Hidden Units:** --hidden_units (default: 512)
- **Dropout Rate:** --dropout (default: 0.2)
- **Epochs:** --epochs (default: 30)
- **Batch Size:** --batch_size (default: 32)
- **Scheduler Parameters:** --step_size (default: 5) and --gamma (default: 0.1)
- **GPU Usage:** Defaults to GPU if available (use --no-gpu to force CPU)

**Example command:** _python train.py --gpu --arch resnet50 --hidden_units 512 --dropout 0.2 --learning_rate 0.003 --weight_decay 5e-5 --epochs 30 --batch_size 32 --step_size 5 --gamma 0.1_

### 📸 Running predict.py

**Purpose:**  
This command loads the saved checkpoint and predicts the flower class for a selected image, prints the top-K predicted classes (with probabilities) and displays a visualization marking whether the very top prediction was the correct one.

Instead of providing an image path, you select the image using:

- **Folder Number:** (--folder_number, default: 1)
- **Image Index:** (--image_index, default: 0)

Additional options include:

- **Top K Predictions:** (--top_k, default: 5)
- **Data Directory:** Defaults to the directory where predict.py lives, with a subfolder flowers.
- **Category Names Mapping:** (--category_names, default: cat_to_name.json)
- **GPU Usage:** Defaults to GPU if available.

**Example command:** _python predict.py final_best_model.pth --folder_number 6 --image_index 3 --top_k 5 --category_names cat_to_name.json --gpu_

## Troubleshooting

- **FileNotFoundError:** Verify that your data directory has the expected structure: it should contain a subfolder named flowers with train, valid, and test.
- **GPU Issues:** If you experience GPU-related issues, try running with --no-gpu to force CPU usage.

## 🤖 Acknowledgment

AI assistance was used for explaining concepts, for refining already written code (chiefly adding comments) and for outputting this README but the code structure, logic, and implementation were independently designed by **cvasilov1**.

-------------------------------------------------------------------------------

## Installation Guide (for first time users)

### Prerequisites

\- **Python 3.7+**

\- **Anaconda (recommended)** for environment management

\- **Git** (to clone the repository)

### Setting Up the Environment

#### 1\. Clone the Repository

Open a terminal and run:

_git clone <https://github.com/your_username/udacity_AIPND_train_flower_network.git>_

_cd udacity_AIPND_train_flower_network_

#### 2\. Set Up a Virtual Environment

**Using Anaconda:**

_conda create --name flower_env python=3.8 -y_

_conda activate flower_env_

**Using venv:**

- On macOS/Linux:

_python3 -m venv flower_env_

_source flower_env/bin/activate_

- On Windows:

_python -m venv flower_env_

_flower_env\\Scripts\\activate_

#### Install Dependencies

If you have a requirements.txt file, run: _pip install -r requirements.txt_

Otherwise, manually install the following packages:

_pip install torch torchvision numpy matplotlib tqdm pillow psutil_
