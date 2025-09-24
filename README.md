# SageMaker Image Classifier with Hyperparameter Tuning

This project demonstrates how to build, tune, and test a deep learning image classifier using AWS SageMaker, TensorFlow/Keras 3, and TFSMLayer for inference. It includes:

1. Cloud-based training and hyperparameter tuning

2. Model deployment and testing with new images

3. Visualizations of predictions and confidence scores

4. A reproducible pipeline for real-world ML workflows

## Project Overview:

Goal: Classify new drawings by two young artists using a convolutional neural network (CNN) trained on their earlier artworks.

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

## Requirements:

AWS SageMaker Studio

TensorFlow/Keras 3

Python 3.8+

Matplotlib, Pandas

## Dataset Summary:

- **Train images used**: 18,813
  - **Artist A**: 10,274 images
  - **Artist B**: 8,539 images
- **Test images used**: 2,908
  - **Artist A**: 1,590 images
  - **Artist B**: 1,318 images

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

| FolderName     | Confidence | Predicted Class |
|:---------------|:----------:|----------------:|
| test_class0_1* |  0.69      |   1             |
| test_class0_2  |  0.89      |   0             |
| test_class0_3  |  0.79      |   0             |
| test_class0_4  |  0.95      |   0             |
| test_class1_1  |  0.65      |   1             |
| test_class1_2  |  0.77      |   1             |
| test_class1_3  |  0.83      |   1             |
| test_class1_4  |  0.77      |   1             |
*First batch was incorrectly predicted due to images' yellow background that the model assosiated with class1 artist. Need more testing to confirm this limitation.

## Image Preprocessing Strategy

To expand the dataset and optimize image resolution, each original artwork was quartered twice, resulting in 16 cropped images per original. This allowed the model to learn from localized features and stylistic details. Note: This method occasionally produced blank or near-blank tiles (e.g., empty corners or borders). These were manually reviewed and removed to ensure clean training data and avoid misleading the model.

## Model upload link:

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