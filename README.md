# PyTorch-Image-Classifier

This project implements an image classifier with PyTorch and provides a command-line application.

## Overview of PyTorch Model


### Dataset
 The dataset the model is trained on comes from the [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
 A custom PyTorch class is devised to facilitate the efficient training, validation, and testing of the PyTorch model. Within this class, training functions are implemented to experiment with different hyperparameters, aiming to attain optimal accuracy.

 The image data undergoes transformation, augmentation, and normalization before being batch-fed into the model. To enhance prediction accuracy and extract image features, the pre-trained MAXVIT_T neural network is employed, followed by a classifier consisting of fully connected linear layers with ReLU activation units and a final softmax layer. Dropout is additionally applied to mitigate overfitting.

The Image Classifier Project.ipynb jupyter notebook file contains the code and steps used for model creation and evaluation.

### Techniques Used
Pytorch
Image normalization/pre-processing
Image Augmentation
Training, Validation and Testing Functions created
Custom Pytorch Class for Model Classifier Implementation
Pre-Trained Neural Networks
Hyperparameter Optimization
Overfitting prevention using Dropout
Image Recognition Inference using a deep learning neural network trained Model

## Files

- **model.py:** Train a new network on the dataset and save the model as a checkpoint.
- **train.py:** Command-line arguments for selecting architectures, setting hyperparameters, and specifying CPU or GPU.
- **predict.py:** Use a trained network to predict the class for an input image.
- **utilities.py:** Helper functions for loading model checkpoints and processing images.
- **cat_to_name.json:** A dictionary mapping integer-encoded categories to actual flower names.


## Accuracy
- Training Accuracy: 81%
- Testing Accuracy: 79%
