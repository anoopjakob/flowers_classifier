# date: 16/07/2020
# programmer : Anoop Jacob

# importing necessary packages

import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

# this place is reserved for linking other py files later
from functions import load_checkpoint, predict, test_model,load_data, process_image
# employing argument parser for controlling program from the command line
import argparse
parser = argparse.ArgumentParser(description='Employing nueral nerwork architecture for prediction of flowers')

# link to provide custom image path for using another image of flower . default path is also defined
parser.add_argument('--image_path', action='store',
                    default = 'flowers/test/30/image_03512',
                    help='Enter a path of image for prediction')
parser.add_argument('--save_dir', action='store',
                    dest='save_directory', default = 'checkpoint.pth',
                    help='Enter location to save the new checkpoint..otherwise will overwrite on previous checkpoint')

parser.add_argument('--arch', action='store',
                    dest='pretrained_model', default='vgg11',
                    help='Enter pretrained model to use, default is VGG-11.')

parser.add_argument('--top_k', action='store',
                    dest='topk', type=int, default = 4,
                    help='Enter number of top most likely predictions to view, default is 4.')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = 'cat_to_name.json',
                    help='Enter path to image.')

parser.add_argument('--gpu', action="store_true", default=False,
                    help='Turn GPU mode on or off, default is = off.')

results = parser.parse_args()

# getting data from the argument parser

save_dir = results.save_directory
image = results.image_path
top_k = results.topk
gpu_mode = results.gpu
cat_names = results.cat_name_dir

# label mapping.. to convert the predictions to actual names of flower
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)


# code to get the original flower name in the image
split_list =image.split('/')
category = split_list[2]





# loading model..
pre_tr_model = results.pretrained_model
model = getattr(models,pre_tr_model)(pretrained=True)
loaded_model = load_checkpoint(model, save_dir, gpu_mode)

# processing image to put it through prediction
processed_image = process_image(image)

# transferring the image to cuda or gpu for efficient working.. else works in cpu also
if gpu_mode == True:
    processed_image = processed_image.to('cuda')
else:
    pass

# prediction
probable_list, classes = predict(processed_image, loaded_model, top_k, gpu_mode)

# Print probabilities and predicted classes
print(probable_list)
print(classes)

# retreiving the name of the flower from cat_to_name dictionary
names = []
for i in classes:
    names += [cat_to_name[i]]

if category in cat_to_name:
    print('The original image is of ',cat_to_name[category])


# Printing the most proabable list
print(f" The flower is probably a: '{names[0]}' with a probability percentage of {round(probable_list[0]*100,2)}% ")

#all predictions are saved in output.txt

# 25 = grape hyacinth
