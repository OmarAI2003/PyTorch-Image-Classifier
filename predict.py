import torch
from torch import nn
from torch import optim
import numpy as np
from torchvision import datasets, transforms, models
import os
import json
import argparse

# Import custom utility functions for processing images and loading checkpoints
from utilities import process_image, load_checkpoint

def main():
    # Function to predict the class of an image using a trained model
    def predict(image_path, model, json_cat, topk=5):
        """ Predict the class (or classes) of an image using a trained deep learning model.
        """
        # Load a JSON file that maps class values to category names
        with open(json_cat, 'r') as f:
            cat_to_name = json.load(f)

        # Get the actual category name from the image path
        lab = cat_to_name[image_path.split('/')[-2]]

        # Load the trained model checkpoint
        with torch.no_grad():
            model = load_checkpoint(model)
            model.eval()
            
            # Process the image for model input
            processed_image = process_image(image_path)

            # Convert the processed image to a PyTorch tensor
            processed_tensor = torch.from_numpy(processed_image).unsqueeze(0).float()

            # Make predictions with the model
            output = torch.exp(model(processed_tensor))
            probs, classes = output.topk(topk, dim=1)
            
            # Map class indices to category names
            classes = [cat_to_name[str(n + 1)] for n in classes.numpy()[0]]
            probs = probs.numpy()[0]

        # Print the predicted probabilities, category names, and the actual name
        print(f"Predicted Probabilities: {probs}")
        print(f"Category Names:         {classes}")
        print(f"Actual Category Name:   {lab}")

    # Define command-line arguments for the script
    parser = argparse.ArgumentParser(description='Arguments for running the model prediction file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for predicting')
    parser.add_argument('--img', help='Enter an image path')
    parser.add_argument('--checkpoint', help="Model's checkpoint file")
    parser.add_argument('--top_k', type=int, default=5, help='Top K probabilities to display')
    parser.add_argument('--categ_names', help='A JSON file that maps class values to category names')
    args = parser.parse_args()

    # Call the predict function with the specified arguments
    predict(image_path=args.img, model=args.checkpoint, json_cat=args.categ_names, topk=args.top_k)

if __name__ == '__main__':
    # Execute the main function when the script is run
    main()
