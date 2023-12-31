# PyTorch-Image-Classifier
In this project, I first developed code for an image classifier built with PyTorch and then converted it into a command-line application.

## The Data
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

## Files
 
 model.py : 
           train a new network on a dataset and save the model as a checkpoint.
 train.py: 
           command line arguments that will allow user to choose (from two different architectures, set hyperparameters for learning rate, number of hidden units, and training epochs, GPU or cpu)
 predict.py: 
           uses a trained network to predict the class for an input image.
 utilities.py: 
           Helper functions used throughout the files to load model checkpoints and process images.
cat_to_name.json: 
           This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

Image Classifier Project: 
           A Jupyter notebook to implement an image classifier with PyTorch.
### Accuracy

### train..81% 
### test..79% 

