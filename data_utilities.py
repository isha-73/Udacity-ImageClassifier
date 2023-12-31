from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data_loaders(data_dir="./flowers"):
    mean, stdev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    #data
    train = datasets.ImageFolder(data_dir + '/train', transform=train_transforms(mean, stdev))
    valid = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms(mean, stdev))
    test= datasets.ImageFolder(data_dir + '/test', transform=test_transforms(mean, stdev))
    class_to_idx = train.class_to_idx
    
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=True)

    return train_loader, valid_loader, test_loader, class_to_idx

def train_transforms(mean, stdev):
    return transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                               transforms.Normalize(mean, stdev)])

def test_transforms(mean, stdev):
    return transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                               transforms.ToTensor(),transforms.Normalize(mean, stdev)])

