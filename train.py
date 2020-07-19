# training section

""" training a new network on a data set with train.py
    prints ourt training loss, validation loss, and validation accuracy as the newtwork trains.

Inputs using argeparse :
   set directory to save checkpoints: train.py data_dir --save_dir directory_name
   choose architecture: train.py --arch "vgg13"
   etc.. """

#importing necessary packages
from time import time
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# importing custom functions
from functions import load_data, build_classifier, validation, train_model, test_model, save_model, load_checkpoint
# employing parser for getting inputs 
parser = argparse.ArgumentParser(description='Train your neural enter or change parameters using argparse commands')

# the data directory is must. all other inputs have default value if not specified
# also remember to turn on gpu by using --gpu during command line call 

# postional argument 
parser.add_argument('data_directory', action= 'store',help ='Enter path of the directory')

#optional arguments
parser.add_argument('--arch', action='store',default='vgg11',dest='pretrained_model',
                    help='enter a pretrained mode. default: vgg11')
parser.add_argument('--save_dir', action='store', default='checkpoint.pth',dest='save_directory',
                    help='enter location to save checkpoint. default: checkpoint.pth')
parser.add_argument('--learning_rate', action='store', dest='lr', type=int, default=0.001,
                    help='Enter learning rate. Default : 0.001')
parser.add_argument('--droput', action='store', dest='drpt', type=int, default=0.05,
                    help='Enter dropout value. Default = 0.05')
parser.add_argument('--hidden_units', action='store', dest='h_units', type=int, default=500,
                    help='Enter the no: of nodes for hidden layer. Default: 500')
parser.add_argument('--epochs', action= 'store', dest='num_epochs', type=int, default=3,
                    help='enter no of epochs. default=3')
parser.add_argument('--gpu', action="store_true",default=False,
                    help='Turn on GPU. Default: OFF')
results = parser.parse_args()

# inputs from parser
data_dir = results.data_directory
save_dir = results.save_directory
learning_rate = results.lr
dropout = results.drpt
hidden_units = results.h_units
epochs = results.num_epochs
gpu_mode = results.gpu

if gpu_mode:
    print('GPU mode is selected')
    if torch.cuda.is_available() is False:
        print('please check the hardware. No GPU is detected')
        while input("Do you want to continue? [y/n]") == "n":
            exit()
else:
    print('Program will train on CPU .. will take longer time')
    while input("Do you want to continue running on CPU mode? [y/n]")=="n":
        exit()
print('... dont switch off computer .. training..')

start = time()

# working on data .. loading processing.. converting it into tensor etc..
trainloader, testloader, validloader, train_data, test_data, valid_data = load_data(data_dir)

# loading a pretrained model
pre_tr_model = results.pretrained_model
model = getattr(models, pre_tr_model)(pretrained = True)

# building a new classifier and replacing it with old classifier
input_units = model.classifier[0].in_features
build_classifier(model, input_units, hidden_units,dropout)

# criterion.. NLLoss selected since we provided logsoftmax for classifier
criterion = nn.NLLLoss()

# optimizer 
optimizer  = optim.Adam(model.classifier.parameters(), learning_rate)

# train model
model, optimizer = train_model(model, epochs,trainloader, validloader, criterion, optimizer, gpu_mode)

# Test model
test_model(model, testloader, gpu_mode)

# Save model
save_model(model, train_data, optimizer, save_dir, epochs)

end = time()
total = end-start
print('!!! Training completed !!! \n \n Total Time Taken: {}:{}:{}'.format( int(total/3600), int((total%3600)/60),int((total%3600)%60) ))
