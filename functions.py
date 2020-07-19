import torch
import numpy as np
from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms

def load_data(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomSizedCrop(224),
                                           transforms.RandomHorizontalFlip(), transforms.ToTensor, 
                                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    # both validation and testing using the same set
    test_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train_data = datasets.ImageFolder(data_dir + '/train', transform = train_transforms)
    test_data = datasets.ImageFolder(datasets + '/test', transform = test_transforms)
    valid_data = datasets.ImageFolder(datasets + '/valid', transform= test_transforms)

    # loading to trainloader and testloader.. only for trainloader shuffle is true
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    return trainloader, testloader, validloader, train_data, test_data, valid_data













