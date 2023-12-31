import torch
import torch.nn as nn
from torchvision import models

import torch
from torchvision import models
import torch.nn as nn

def build_network(arch, output_dim, hidden_dim, drop_prob):
   
    """
    Build and return a pre-trained CNN model based on the specified architecture.

    Args:
    - arch (str): Architecture name, e.g., 'vgg16', 'resnet18', 'densenet121'.
    - output_dim (int): Number of output nodes in the final layer.
    - hidden_dim (int): Number of nodes in the hidden layer.
    - drop_prob (float): Dropout probability.

    Returns:
    - model (torch.nn.Module): Pre-trained CNN model with a modified classifier.
    """

    # to check if the specified architecture is supported
    supported_arch= ['vgg16', 'resnet18', 'densenet121']
    if arch not in supported_arch:
        raise ValueError(f"Unsupported architecture. Supported architectures: {supported_architectures}")
        
    print(f"Using {arch}")
    # Load the pre-trained model based on the specified architecture
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[0].in_features
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features

    # Freeze parameters to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False

    # Define the new classifier
    classifier = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, output_dim),
        nn.LogSoftmax(dim=1)
    )

    # Replace the original classifier with the new one
    if arch == 'vgg16':
        model.classifier = classifier
    elif arch == 'resnet18':
        model.fc = classifier
    elif arch == 'densenet121':
        model.classifier = classifier

    # Move the model to the GPU
    model.cuda()

    return model



def save_model(model, arch, epochs, lr, hidden_units):
   
    save_path = f'./checkpoint_{arch}.pth'
    checkpoint = {
        'arch': arch,
        'hidden_dim': hidden_units,
        'epochs': epochs,
        'lr': lr,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, save_path)

def load_model(checkpoint):
    trained = torch.load(checkpoint)
    model = build_network(arch=trained['arch'],
                          output_dim=102, hidden_dim=trained['hidden_dim'], drop_prob=0)

    model.class_to_idx = trained['class_to_idx'] # configuring class to index
    model.load_state_dict(trained['state_dict'])
    print(f"Model has been loaded with {trained['arch']} architecture")
    return model