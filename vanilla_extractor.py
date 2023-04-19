import torch
import os
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import *
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch.autograd import Variable
from torchvision.models import ResNet50_Weights
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from typing import Type, Any, Callable, Union, List, Dict, Optional, cast
from torch import Tensor
from collections import OrderedDict 
# from torchsummary import summary

import pandas as pd

def get_transform(dataset):
    transform = None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return transform

class SaveOutput:
    # stores the hooked layers
    # Source 1: https://towardsdatascience.com/the-one-pytorch-trick-which-you-should-know-2d5e9c1da2ca
    # Source 2: https://discuss.pytorch.org/t/extracting-stats-from-activated-layers-training/45880/2
    def __init__(self):
        self.outputs = OrderedDict()

    def save_activation(self, name):
        def hook(module, module_in, module_out):
            self.outputs.update({name: module_out})

        return hook

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

class Resnet50_FX(nn.Module):

    REGION_LIST = ['B1U1', 'B1U2', 'B1U3',
                   'B2U1', 'B2U2', 'B2U3', 'B2U4',
                   'B3U1', 'B3U2', 'B3U3', 'B3U4', 'B3U5', 'B3U6',
                   'B4U1', 'B4U2', 'TCL',
                   'POOL', 'FC1']

    def __init__(self, dataset="ImageNet", device="cpu"):
        super().__init__()
        self.device = device

        # regions we'll record from
        self.hook_map = OrderedDict([
            (Resnet50_FX.REGION_LIST[0], ["layer1", 0]),
            (Resnet50_FX.REGION_LIST[1], ["layer1", 1]),
            (Resnet50_FX.REGION_LIST[2], ["layer1", 2]),
            (Resnet50_FX.REGION_LIST[3], ["layer2", 0]),
            (Resnet50_FX.REGION_LIST[4], ["layer2", 1]),
            (Resnet50_FX.REGION_LIST[5], ["layer2", 2]),
            (Resnet50_FX.REGION_LIST[6], ["layer2", 3]),
            (Resnet50_FX.REGION_LIST[7], ["layer3", 0]),
            (Resnet50_FX.REGION_LIST[8], ["layer3", 1]),
            (Resnet50_FX.REGION_LIST[9], ["layer3", 2]),
            (Resnet50_FX.REGION_LIST[10], ["layer3", 3]),
            (Resnet50_FX.REGION_LIST[11], ["layer3", 4]),
            (Resnet50_FX.REGION_LIST[12], ["layer3", 5]),
            (Resnet50_FX.REGION_LIST[13], ["layer4", 0]),
            (Resnet50_FX.REGION_LIST[14], ["layer4", 1]),
            (Resnet50_FX.REGION_LIST[15], ["layer4", 2]),
            (Resnet50_FX.REGION_LIST[16], ["avgpool"]),
            (Resnet50_FX.REGION_LIST[17], ["fc"])
        ])

        self.save_output = SaveOutput()

        self.model = models.resnet50().to(device)


        self.model.load_state_dict(torch.load("/Users/matt/Desktop/NSCI Project/pytorch tutorials/EIG-Bodies-master/ResNet50-ImageNet.pth"))

        self.transform = get_transform(dataset)

        for region in self.hook_map:
            hook_loc = self.hook_map[region]
            layer = getattr(self.model, hook_loc[0])
            if len(hook_loc) == 2:
                layer = layer[hook_loc[1]]
            layer.register_forward_hook(self.save_output.save_activation(region))

    def forward(self, x):
        x = self.transform(x).to(self.device).unsqueeze(0)
        self.model(x)

        return self.save_output.outputs


model = Resnet50_FX()

index = 0
images = 'images'
df = pd.DataFrame(columns=['id', 'title', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16']) # Column names
flag = 0

for filename in os.listdir(images):
    
    f = os.path.join(images, filename)
    if os.path.isfile(f) and f.split('/')[1].split('_')[0] != '.DS':
        print("Working on " + str(filename))
        img = Image.open(f).convert('RGB')
        #pre_img = preprocess(img)
        #tensor_img = torch.unsqueeze(pre_img, 0)
        model(img)
        array = list(model.save_output.outputs.values())

        l1 = array[0].flatten().tolist()
        l2 = array[1].flatten().tolist()
        l3 = array[2].flatten().tolist()
        l4 = array[3].flatten().tolist()
        l5 = array[4].flatten().tolist()
        l6 = array[5].flatten().tolist()
        l7 = array[6].flatten().tolist()
        l8 = array[7].flatten().tolist()
        l9 = array[8].flatten().tolist()
        l10 = array[9].flatten().tolist()
        l11 = array[10].flatten().tolist()
        l12 = array[11].flatten().tolist()
        l13 = array[12].flatten().tolist()
        l14 = array[13].flatten().tolist()
        l15 = array[14].flatten().tolist()
        l16 = array[15].flatten().tolist()

        # checking if it is a file
        if os.path.isfile(f) and f.split('/')[1].split('_')[0] != '.DS' and f.split('/')[1].split('.')[0].split('_')[1] != 'light-diff' and f.split('/')[1].split('.')[0].split('_')[1] != 'posture-diff':
            if f.split('/')[1].split('.')[0].split('_')[2] == 'mooney':
                df.loc[index] = [f.split('/')[1].split('_')[0], f.split('/')[1].split('.')[0], l1, l2, l3, l4, l5, l6 ,l7, l8, l9, l10, l11, l12, l13, l14, l15, l16]

                index = index + 1
    
                print("Image finished: " + str(index))
                #summary(model,input_size=(3, 224, 224))

product = df.sort_values(['title'], ascending=[True]) # alphabetize values
product.to_csv('mooney_features_vanilla.csv') # save df
print("done")