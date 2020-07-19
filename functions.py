import torch
import pandas as pd
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
def validation(model, validloader, criterion, gpu_mode):
    valid_loss = 0
    accuracy = 0

    # checking for gpu mode
    if gpu_mode == True:
        model.to('cuda')
    else:
        pass
    # data from valid loader used for validation
    for ii ,(images, labels) in enumerate(validloader):
        if gpu_mode == True:
        # change model to work with cuda
            if torch.cuda.is_available():
                images, labels = images.to('cuda'), labels.to('cuda')
        else:
            pass
        #forward pass
        output = model.forward(images)

        #calculating loss
        valid_loss += criterion(output, labels).item()
        # calcualting probability
        ps = torch.exp(output) #because output is in log form
        #calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy+= equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy 


def train_model(model , epochs, trainloader, validloader, criterion, optimizer, gpu_mode):
    # epochs =3
    iterations = 0
    print_every = 10
    if gpu_mode == True:
        model.to('cuda')
    else:
        pass
    for e in range(epochs):
        running_loss = 0
        # getting values for iteration from trainloader
        for ii, (inputs, labels) in enumerate(trainloader):
            iterations+=1

            if gpu_mode == True:
                inputs, labels == inputs.to('cuda'), labels.to('cuda')
            else:
                pass
            
            # parameter gradinets set to zero ..
            optimizer.zero_grad()

            # forward pass
            outputs = model.forward(inputs)

            # calcuating the model loss.. variation from the actual value
            loss = criterion(outputs,labels)

            # backward pass
            loss.backward()
            
            #updating the weights
            optimizer.step()

            # adding all losses to calculate the training loss
            running_loss+= loss

            # validation area
            if iterations % print_every == 0:
                # evaluation mode.. 
                model.eval()
            #here backpropogation or gradient upadation not required
                with torch.no_grad():
                    # using validaation function..
                    valid_loss, accuracy = validation(model, validloader, criterion, gpu_mode)
                print(f"No. epochs: {e+1}, \
                    Training Loss: {round(running_loss/print_every,3)} \
                    Valid Loss: {round(valid_loss/len(validloader),3)} \
                    Valid accuracy: {round(float(accuracy/len(validloader)),3)}")
                running_loss = 0
                
                #Resuming training
                model.train()
    return model, optimizer









