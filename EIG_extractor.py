import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from collections import OrderedDict
from os.path import *

import numpy as np
import pandas as pd
import os
from PIL import Image

from smplx import SMPL as _SMPL
from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints
from utils import smpl_constants
from utils.geometry import rot6d_to_rotmat


CAM = "CAMERA"
DEC_GLOBORIENT = "DEC_GORIENT"
DEC_POSE = "DEC_POSE"
DEC_SHAPE = "DEC_SHAPE"
DEC_CAM = "DEC_CAM"
PRED_GLOBORIENT = "PRED_GORIENT"
PRED_POSE = "PRED_POSE"
PRED_SHAPE = "PRED_SHAPE"
PRED_CAM = "PRED_CAM"
VERTICES = "VERTICES"
VERTICES_NR = "VERTICES_NR"
SMPL_GLOBORIENT = "SMPL_GORIENT"
SMPL_POSE = "SMPL_POSE"
SMPL_JOINTS = "SMPL_JOINTS"
SMPL_BETAS = "SMPL_BETAS"
SMPL_PB = "SMPL_POSE&BETAS"

SMPL_MEAN_PARAMS = '/Users/matt/Desktop/NSCI Project/pytorch tutorials/EIG-Bodies-master/smpl_mean_params.npz'

def get_transform(dataset):
    transform = None
    if dataset == "ImageNet" or dataset == "Random":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        print("Not implemented.")

    return transform


class Paths:
    SMPL_MEAN_PARAMS = join(dirname(abspath(__file__)), '/Users/matt/Desktop/NSCI Project/pytorch tutorials/EIG-Bodies-master/smpl_mean_params.npz')


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

        if dataset == "ImageNet":
            self.model.load_state_dict(torch.load(Paths.RESNET_IMAGENET))
        elif dataset == "Places":
            self.model.load_state_dict(torch.load(Paths.RESNET_PLACES))
        elif dataset == "Random":
            pass
        else:
            print("Not implemented.")

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


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [smpl_constants.JOINT_MAP[i] for i in smpl_constants.JOINT_NAMES]
        j_regressor_extra = np.load(Paths.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(j_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

        self.smpl_joints = None

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)

        self.smpl_joints = smpl_output.joints

        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output


class _HMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    REGION_LIST = ['B1U1', 'B1U2', 'B1U3',
                   'B2U1', 'B2U2', 'B2U3', 'B2U4',
                   'B3U1', 'B3U2', 'B3U3', 'B3U4', 'B3U5', 'B3U6',
                   'B4U1', 'B4U2', 'TCL', 'POOL', 'FC1', 'FC2',
                   'CAMERA', 'VERTICES', 'GLOBAL_ORIENT', 'BODY_POSE', 'JOINTS', 'BETAS',
                   'VERTICES_NOROT', 'SMPL_POSE']

    def __init__(self, device="cpu", no_ief=False):
        super().__init__()
        self.device = device
        self.no_ief = no_ief

        n_pose = 24 * 6

        # self.smpl_model = smpl_model

        # regions we'll record from
        self.hook_map = OrderedDict([
            (_HMR.REGION_LIST[0], ["layer1", 0]),
            (_HMR.REGION_LIST[1], ["layer1", 1]),
            (_HMR.REGION_LIST[2], ["layer1", 2]),
            (_HMR.REGION_LIST[3], ["layer2", 0]),
            (_HMR.REGION_LIST[4], ["layer2", 1]),
            (_HMR.REGION_LIST[5], ["layer2", 2]),
            (_HMR.REGION_LIST[6], ["layer2", 3]),
            (_HMR.REGION_LIST[7], ["layer3", 0]),
            (_HMR.REGION_LIST[8], ["layer3", 1]),
            (_HMR.REGION_LIST[9], ["layer3", 2]),
            (_HMR.REGION_LIST[10], ["layer3", 3]),
            (_HMR.REGION_LIST[11], ["layer3", 4]),
            (_HMR.REGION_LIST[12], ["layer3", 5]),
            (_HMR.REGION_LIST[13], ["layer4", 0]),
            (_HMR.REGION_LIST[14], ["layer4", 1]),
            (_HMR.REGION_LIST[15], ["layer4", 2]),
            (_HMR.REGION_LIST[16], ["avgpool"]),
            (_HMR.REGION_LIST[17], ["FC1"]),
            (_HMR.REGION_LIST[18], ["FC2"]),
            (DEC_CAM, ["deccam"]),
            (DEC_POSE, ["decpose"]),
            (DEC_SHAPE, ["decshape"]),
        ])

        self.transform = get_transform("ImageNet")
        self.save_output = SaveOutput()

        self.resnet = nn.Sequential()
        for name, child in models.resnet50().named_children():
            if name != "fc":
                self.resnet.add_module(name, child)

        if self.no_ief:
            self.fc1 = nn.Linear(512 * 4, 1024)
        else:
            self.fc1 = nn.Linear(512 * 4 + n_pose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, n_pose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        for region in self.hook_map:
            if region == "FC1":
                layer = self.fc1
            elif region == "FC2":
                layer = self.fc2
            elif "DEC" in region:
                layer = getattr(self, self.hook_map[region][0])
            else:
                hook_loc = self.hook_map[region]
                layer = getattr(self.resnet, hook_loc[0])
                if len(hook_loc) == 2:
                    layer = layer[hook_loc[1]]
            layer.register_forward_hook(self.save_output.save_activation(region))

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        x = self.transform(x).to(self.device).unsqueeze(0)

        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.resnet(x)
        xf = x.view(x.size(0), -1)

        if self.no_ief:
            # regress latents directly
            print("Using no IEF this time")
            xc = self.fc1(xf)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc)
            pred_shape = self.decshape(xc)
            pred_cam = self.deccam(xc)
        else:
            pred_pose = init_pose
            pred_shape = init_shape
            pred_cam = init_cam
            for i in range(n_iter):
                xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
                xc = self.fc1(xc)
                xc = self.drop1(xc)
                xc = self.fc2(xc)
                xc = self.drop2(xc)
                pred_pose = self.decpose(xc) + pred_pose
                pred_shape = self.decshape(xc) + pred_shape
                pred_cam = self.deccam(xc) + pred_cam

        pred_rotation_mat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        self.save_output.outputs[DEC_GLOBORIENT] = self.save_output.outputs[DEC_POSE].view(-1, 3, 2)[0]
        self.save_output.outputs[DEC_POSE] = self.save_output.outputs[DEC_POSE].view(-1, 3, 2)[1:]
        self.save_output.save_activation(PRED_GLOBORIENT)(None, None, pred_pose.view(-1, 3, 2)[0])
        self.save_output.save_activation(PRED_POSE)(None, None, pred_pose.view(-1, 3, 2)[1:])
        self.save_output.save_activation(PRED_SHAPE)(None, None, pred_shape)
        self.save_output.save_activation(PRED_CAM)(None, None, pred_cam)
        
        return self.save_output.outputs


class HmrResnet50_FX(_HMR):
    def __init__(self, human=True, state_dict=None, device="cpu", no_ief=False):
        super().__init__(SMPL_MEAN_PARAMS,
                         SMPL(Paths.SMPL_MODEL_DIR,
                              batch_size=1,
                              create_transl=False).to(device),
                         device=device,
                         no_ief=no_ief
                         )

        if human:
            pre_trained_state_dict = torch.load(Paths.RESNET_SPIN, map_location=device)['model']
        else:
            pre_trained_state_dict = torch.load(Paths.RESNET_SPIN_MONKEY, map_location=device)['model']
        if state_dict is not None:
            pre_trained_state_dict = state_dict

        modified_dict = {}
        for k, v in pre_trained_state_dict.items():
            if k.startswith('conv') or k.startswith('layer') or k.startswith('bn'):
                modified_dict['resnet.' + k] = v
            else:
                modified_dict[k] = v

        self.load_state_dict(modified_dict, strict=False)

    def get_smpl_model(self):
        return self.smpl_model

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

def main():
    model = _HMR()
    
    index = 0
    images = 'images'
    df = pd.DataFrame(columns=['id', 'title', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16']) # Column names
    flag = 0

    for filename in os.listdir(images):
    
        f = os.path.join(images, filename)
        if os.path.isfile(f) and f.split('/')[1].split('_')[0] != '.DS':
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
    
    product = df.sort_values(['title'], ascending=[True]) # alphabetize values
    product.to_csv('mooney_features_EIG.csv') # save df
    print("done")

if __name__ == '__main__':
    main()
