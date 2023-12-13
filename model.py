import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
import os 



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    


def train_model(gpu, model_name, hidden_units, data_dir , save_dir, epochs=1, learning_rate=0.0001):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Defining  transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(225), transforms.CenterCrop(224),transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])


    # Loading the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)


    # Using the image datasets and the trainforms to  define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
  
    
    # Model name to class mapping
    model_name_to_class = {
        'vgg13': models.vgg13(pretrained=True),
        'alexnet': models.alexnet(pretrained=True)}
    
        # Create the model based on the model name
    model = model_name_to_class[model_name]

        
    

    # Freeze parameters so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(hidden_units, 4096),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(4096, 102),   # Output layer
        nn.LogSoftmax(dim=1)
    )

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Use GPU if available
    device = ("cuda" if torch.cuda.is_available() and gpu else "cpu")

    model.to(device)

    steps = 0
    running_loss = 0
    print_every=5
    try:
        for epoch in range(epochs):
            for images, labels in trainloaders:
                steps += 1

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()

                    with torch.no_grad():
                        for images, labels in validloaders:
                            images, labels = images.to(device), labels.to(device)

                            logps = model(images)
                            batch_loss = criterion(logps, labels)
                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch + 1}/{epochs}.. "
                          f"train loss {running_loss / print_every:.3f}..."
                          f"validation loss {valid_loss / len(validloaders):.3f}..."
                          f" Accuracy: {accuracy / len(validloaders):.3f}")
                    running_loss = 0
                    model.train()

        # Save the model and additional information to a checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epochs': epochs,
            'model_architecture': model_name,
            'learning_rate': learning_rate,
            'hidden_units': hidden_units}

        # Specify the file path to save the checkpoint
        torch.save(checkpoint, save_dir)

    except KeyboardInterrupt:
        print("Training interrupted.")
