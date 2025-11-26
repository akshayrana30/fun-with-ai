# AI/ML Models in fun-with-ai Repository

This document lists all the machine learning models and implementations found in this repository.

## Overview

This repository contains various AI/ML practice projects using different frameworks and algorithms. Use the `list_models.py` script to automatically discover and list all models in the repository.

## Usage

To list all models in the repository, run:

```bash
python3 list_models.py
```

This will scan the repository and display:
- Saved model files (*.h5, *.pkl, *.pt, etc.)
- Python files containing model implementations
- Model architectures and frameworks used
- Summary statistics

## Model Categories

### 1. Convolutional Neural Networks (CNN)
**Location:** `CNN/`

- **image_classification.py** - Image classification using Keras Sequential CNN
  - Framework: Keras/TensorFlow
  - Architecture: Multiple Conv2D layers with MaxPooling
  - Use case: Binary classification of images (Flowers vs Animals)

### 2. Classifiers
**Location:** `Classifier/`

- **SVM.py** - Support Vector Machine classifier
  - Framework: scikit-learn
  - Algorithm: SVM with SVC
  
- **Gaussion NB.py** - Gaussian Naive Bayes classifier
  - Framework: scikit-learn
  - Algorithm: Gaussian Naive Bayes
  
- **Trees.py** - Decision Tree classifier
  - Framework: scikit-learn
  - Algorithm: Decision Trees

### 3. TensorFlow Examples
**Location:** `Tensors/`

- **start_contrib.py** - TensorFlow contrib module examples
- **GradientDescent-contrib.py** - Gradient descent using TensorFlow contrib
- **GradientDescent-core.py** - Gradient descent using TensorFlow core
- **Sentiment_Analysis.py** - Sentiment analysis implementation
- **Customized-Estimator.py** - Custom TensorFlow estimator
- **rnn_tut.py** - Recurrent Neural Network tutorial
- **tfLearn.py** - TFLearn examples
- **intro-to-tensors.py** - Introduction to TensorFlow tensors

### 4. Self-Driving Car
**Location:** `Self-Driving/`

- **model.py** - NVIDIA architecture for behavioral cloning
  - Framework: Keras/TensorFlow + scikit-learn
  - Architecture: CNN with 5 convolutional layers and 4 fully connected layers
  - Functions: build_model(), train_model(), load_data()
  - Use case: Steering angle prediction for self-driving cars
  
- **drive.py** - Driver script to use trained model

### 5. Reinforcement Learning
**Location:** `Reinforcement/`

- **Ping Pong/main.py** - Reinforcement learning for Pong game
- **demo.py** - RL demonstration
- **World/main.py** - World simulation for RL

### 6. Clustering
**Location:** `Clustering/`

- **k-means/kmeans.py** - K-means clustering implementation
- **k-means/execute.py** - K-means execution script
  - Framework: scikit-learn

### 7. Computer Vision
**Location:** `Webcam face detect/`

- **webcam_cv3.py** - Webcam face detection using OpenCV

### 8. Embeddings
**Location:** `Embeddings/`

- **Thrones2Vec.ipynb** - Word embeddings for Game of Thrones text
- **demo.ipynb** - Embeddings demonstration

## Jupyter Notebooks

- **Fashion MNIST using ML & DL.ipynb** - Fashion MNIST classification
- **PCA.ipynb** - Principal Component Analysis examples

## Framework Summary

The repository uses the following frameworks:
- **TensorFlow** - Deep learning framework
- **Keras** - High-level neural networks API
- **scikit-learn** - Machine learning library
- **OpenCV** - Computer vision library
- **PyTorch** (if any models use it)

## Adding New Models

When adding new models to the repository:
1. Place them in appropriate category folders
2. Include clear comments about the architecture
3. Run `python3 list_models.py` to verify they're detected
4. Update this MODELS.md file if needed

## Model Discovery Tool

The `list_models.py` script automatically:
- Scans for saved model files (*.h5, *.pkl, *.pt, *.pth, etc.)
- Identifies model implementations in Python files
- Detects frameworks used (TensorFlow, PyTorch, scikit-learn)
- Identifies model types (CNN, RNN, SVM, etc.)
- Counts approximate number of layers
- Lists model-related functions

Run it regularly to keep track of all models in the repository!
