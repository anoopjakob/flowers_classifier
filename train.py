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

# employing parser for getting inputs 

parser = argparse.ArgumentParser(description='Train your neural enter or change parameters using argparse commands')