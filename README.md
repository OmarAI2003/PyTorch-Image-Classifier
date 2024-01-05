# PyTorch-Image-Classifier

This project implements an image classifier with PyTorch and provides a command-line application.

## Dataset
Download the dataset from [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

## Files

- **model.py:** Train a new network on the dataset and save the model as a checkpoint.
- **train.py:** Command-line arguments for selecting architectures, setting hyperparameters, and specifying CPU or GPU.
- **predict.py:** Use a trained network to predict the class for an input image.
- **utilities.py:** Helper functions for loading model checkpoints and processing images.
- **cat_to_name.json:** A dictionary mapping integer-encoded categories to actual flower names.

## Image Classifier Project
A Jupyter notebook for implementing the image classifier with PyTorch.

### Accuracy
- Training Accuracy: 81%
- Testing Accuracy: 79%
