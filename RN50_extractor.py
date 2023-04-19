import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from collections import OrderedDict
from os.path import *
import clip
import torch
import clip
from numpy import load
import pandas as pd
import os
from IPython.display import display, HTML
from PIL import Image
from torchvision.models.feature_extraction import get_graph_node_names

import numpy as np

from smplx import SMPL as _SMPL
from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints
from utils import smpl_constants
from utils.geometry import rot6d_to_rotmat

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

class CLIP(nn.Module):

    REGION_LIST = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16']

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        # regions we'll record from
        self.hook_map = OrderedDict([
            (CLIP.REGION_LIST[0], ["layer1", 0]),
            (CLIP.REGION_LIST[1], ["layer1", 1]),
            (CLIP.REGION_LIST[2], ["layer1", 2]),
            (CLIP.REGION_LIST[3], ["layer2", 0]),
            (CLIP.REGION_LIST[4], ["layer2", 1]),
            (CLIP.REGION_LIST[5], ["layer2", 2]),
            (CLIP.REGION_LIST[6], ["layer2", 3]),
            (CLIP.REGION_LIST[7], ["layer3", 0]),
            (CLIP.REGION_LIST[8], ["layer3", 1]),
            (CLIP.REGION_LIST[9], ["layer3", 2]),
            (CLIP.REGION_LIST[10], ["layer3", 3]),
            (CLIP.REGION_LIST[11], ["layer3", 4]),
            (CLIP.REGION_LIST[12], ["layer3", 5]),
            (CLIP.REGION_LIST[13], ["layer4", 0]),
            (CLIP.REGION_LIST[14], ["layer4", 1]),
            (CLIP.REGION_LIST[15], ["layer4", 2])
        ])
        self.save_output = SaveOutput()

        # load the model
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("RN50", device=device)
        self.transformer = self.model.visual
        #self.model.eval()

        for region in self.hook_map:
            hook_loc = self.hook_map[region]
            layer = getattr(self.transformer, str(hook_loc[0]))[hook_loc[1]]
            layer.register_forward_hook(self.save_output.save_activation(region))

    def forward(self, x):
        x = self.transform(x).to(self.device).unsqueeze(0)
        self.model(x)

        return self.save_output.outputs

c = CLIP()

index = 0
images = 'images'
df = pd.DataFrame(columns=['id', 'title', 'local shape', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'global shape', 'global features', 'model']) # Column names
flag = 0

for filename in os.listdir(images):
    
    f = os.path.join(images, filename)
    if os.path.isfile(f) and f.split('/')[1].split('_')[0] != '.DS':
        image = c.preprocess(Image.open(f)).unsqueeze(0).to("cpu")
        text = clip.tokenize(["a diagram", "a dog", "a cat", "a snake"]).to("cpu")

        with torch.no_grad():
            image_features = c.model.encode_image(image)
            text_features = c.model.encode_text(text)
    
            logits_per_image, logits_per_text = c.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        array = list(c.save_output.outputs.values())

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
    if os.path.isfile(f) and f.split('/')[1].split('_')[0] != '.DS':
        df.loc[index] = [f.split('/')[1].split('_')[0], f.split('/')[1].split('.')[0], array[0].shape, l1, l2, l3, l4, l5, l6 ,l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, "NA", "NA", "RN50"]

        index = index + 1
    
        print("Image finished: " + str(index))
    

product = df.sort_values(['title'], ascending=[True]) # alphabetize values
product.to_pickle('mooney_features_RN50.pkl') # save df








