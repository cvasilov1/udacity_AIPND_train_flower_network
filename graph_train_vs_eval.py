# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:13:49 2025

@author: camel
"""

import matplotlib.pyplot as plt

# Sample data (Replace with real loss values)
epochs = list(range(1, 21))  # Example for 20 epochs
train_losses = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.43, 0.41, 0.4, 0.39, 0.38, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.3, 0.29]
val_losses = [0.95, 0.85, 0.75, 0.67, 0.6, 0.58, 0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41]

# Plot Loss Graph
plt.figure(figsize=(8,5))
plt.plot(epochs, train_losses, label="Training Loss", marker="o")
plt.plot(epochs, val_losses, label="Validation Loss", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
