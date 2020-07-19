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

# loading and processing image to use in the final stage of prediction
# this function gets a image path and converts it into tensor and then to numpy array of optimum requirements
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # DONE: Process a PIL image for use in a PyTorch model

    # Converting image to PIL image
    pil_im = Image.open(f'{image}' + '.jpg')

    # transforming
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    pil_tfd = transform(pil_im)
    
    # Converting to Numpy array 
    array_im_tfd = np.array(pil_tfd)
    
    #return array_im_tfd
    
    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(array_im_tfd).type(torch.FloatTensor)
    # Adding dimension 
    img_add_dim = img_tensor.unsqueeze_(0)
    
    return img_add_dim

def test_model(model, testloader, gpu_mode):
    correct = 0
    total = 0
    
    if gpu_mode == True:
        model.to('cuda')
    else:
        pass

    with torch.no_grad():
        # loading iterables from the test loader
        for ii, (images, labels) in enumerate(testloader):
            
            if gpu_mode == True:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Test accuracy of model for {total} images: {round(100 * correct / total,3)}%")


    
def save_model(model, train_data, optimizer, save_dir, epochs):
    
    checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'opt_state': optimizer.state_dict,
                  'num_epochs': epochs}

    return torch.save(checkpoint, save_dir)



def load_checkpoint(model, save_dir, gpu_mode):
    
    
    if gpu_mode == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)
    
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
            
    
   
def predict(processed_image, loaded_model, topk, gpu_mode):
    
  
    # evaluation mode
    loaded_model.eval()
    
    if gpu_mode == True:
        loaded_model.to('cuda')
    else:
        loaded_model.cpu()
    
    # turning gradient off
    with torch.no_grad():
        # Running image through network
        output = loaded_model.forward(processed_image)

    # Calculating probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    # Converting it into np array or lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    # Loading index and class mapping
    class_to_idx = loaded_model.class_to_idx
    
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

 
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list
