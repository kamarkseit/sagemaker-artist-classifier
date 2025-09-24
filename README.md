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

```
sagemaker-artist-classifier/
├── README.md
├── train.py
├── Launcher.ipynb
├── inference_tests.ipynb
├── test_images/
└── utils/
    └── quarter_images.ipynb
```

## Tests Results:

| FolderName   | Confidence | Predicted Class |
|:-------------|:----------:|----------------:|
| test_class0* |  0.69      |   1             |
| test_class0  |  0.89      |   0             |
| test_class0  |  0.79      |   0             |
| test_class0  |  0.95      |   0             |
| test_class1  |  0.65      |   1             |
| test_class1  |  0.77      |   1             |
| test_class1  |  0.83      |   1             |
| test_class1  |  0.77      |   1             |
*First batch was incorrectly predicted due to images' yellow background that the model assosiated with class1 artist.

## Requirements:

AWS SageMaker Studio

TensorFlow/Keras 3

Python 3.8+

Matplotlib, Pandas

## Model upload S3 link:

https://sagemaker-artist-classifier.s3.us-east-2.amazonaws.com/model.tar.gz

## Further Improvements:

### Model Hosting and Deployment

Set up a real-time SageMaker endpoint for live predictions

Add a Flask or FastAPI wrapper for local serving and integration

### Enhanced Model Architecture

Experiment with transfer learning using pretrained models (e.g., ResNet, EfficientNet)

### Evaluation and Metrics

Include precision, recall, F1-score, and confusion matrix visualizations

Compare performance across tuning jobs and log results

### Dataset Expansion

Add more diverse artwork categories or styles

Incorporate metadata (e.g., artist, medium, era) for multi-label classification