import torch
import numpy as np
from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from collections import OrderedDict

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

def build_classifier(model, input_units, hidden_units, dropout):
    # freezing the weights of the pretrained model
    for param in model.parameters():
        param.requires_grad = False
    # building new classifier section only
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_units, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_units,102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    # replacing the original classifier with the new one
    model.classifier = classifier
    return model











