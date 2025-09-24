# SageMaker Image Classifier with Hyperparameter Tuning

This project demonstrates how to build, tune, and test a deep learning image classifier using AWS SageMaker, TensorFlow/Keras 3, and TFSMLayer for inference. It includes:

1. Cloud-based training and hyperparameter tuning

2. Model deployment and testing with new images

3. Visualizations of predictions and confidence scores

4. A reproducible pipeline for real-world ML workflows

## Project Overview:

Goal: Classify images using a CNN trained on custom data

Platform: AWS SageMaker Studio (JupyterLab)

Framework: TensorFlow/Keras 3

Tuning: SageMaker HyperparameterTuner

Inference: Keras TFSMLayer for SavedModel format

## Key Features:

-Hyperparameter tuning with custom ranges

-Inference testing on new images

-Confidence score visualizations

-Memory-safe batch prediction

-Organized repo for reproducibility

## Repository Structure:

├── README.md
├── train.py                  # Model architecture and training logic
├── Launcher.ipynb               # Launches tuning job in SageMaker
├── inference_tests.ipynb        # Loads model and runs predictions
├── model/                       # Extracted saved model
│   └── model.tar.gz
├── test_images/                 # Folder of images for testing
└── utils/
    └── quarter_images.ipynb         # Image quartering function

## Tests Results:

| FolderName   | Confidence | Predicted Class |
|:-------------|:----------:|----------------:|
| test_class0  |  0.69      |   1             |
| test_class0  |  0.89      |   0             |
| test_class0  |  0.79      |   0             |
| test_class0  |  0.95      |   0             |
| test_class1  |  0.65      |   1             |
| test_class1  |  0.77      |   1             |
| test_class1  |  0.83      |   1             |
| test_class1  |  0.77      |   1             |
First batch was incorrectly predicted due to images' yellow background that the model assosiated with class1 artist.

## Requirements:

AWS SageMaker Studio

TensorFlow/Keras 3

Python 3.8+

Matplotlib, Pandas

## Model upload S3 link:

