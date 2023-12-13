import torch
from torch import nn
from torch import optim
import numpy as np
from torchvision import datasets, transforms, models
import os
from PIL import Image

# Function to load a model checkpoint from a file
def load_checkpoint(filepath):
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location=("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Get the model architecture name from the checkpoint
    model_name = checkpoint['model_architecture']
    
    # Create the model with the same architecture
    model = getattr(models, model_name)(pretrained=True)
    
    # Get the number of hidden units from the checkpoint
    hidden_units = checkpoint['hidden_units']
    
    # Define a custom classifier for the model
    classifier = nn.Sequential(
        nn.Linear(hidden_units, 4096),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(4096, 102),   # Output layer
        nn.LogSoftmax(dim=1)
    )
    
    # Replace the model's classifier with the custom one
    model.classifier = classifier
    
    # Load the model's state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

# Function to process an image for model input
def process_image(image):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model, returns a NumPy array
    '''
    # Open and convert the image to RGB format
    img = Image.open(image).convert('RGB')
    
    # Apply a series of transformations to the image
    img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(img)
    
    return img.numpy()
