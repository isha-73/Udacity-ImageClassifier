import torch
import torch.nn as nn
from torchvision import models

def build_network(arch, output_dim, hidden_dim, drop_prob):
    # loading pre-trained VGG16 model
    model = models.vgg16(pretrained=True)  
    
    # freeze parameters to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False
    
    # defining the new classifier
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, output_dim),
        nn.LogSoftmax(dim=1)
    )
    
    # Replace the original classifier with the new one
    model.classifier = classifier
    
    # Move the model to the GPU
    model.cuda()
    
    return model


def save_model(model, arch, epochs, lr, hidden_units):
   
    save_path = f'./checkpoint-{arch}.pth'
    
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