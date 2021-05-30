import sys, os
sys.path.append(os.path.abspath(os.path.join('./models')))
sys.path.append(os.path.abspath(os.path.join('./Data_Loaders')))
sys.path.append(os.path.abspath(os.path.join('./Loss_Functions')))

import pytorch_lightning as pl

from trainer import CoolSystem

from arch import SegNet, SegNet_Split
from arch_2 import Depth_Net
from resnet import ResNetUNet, ResNetUNet50
from densenet_2 import FCDenseNet103
import data_loader
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms, utils
# import config
import torch
import tqdm
# import loss_functions
import gc
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image



if __name__ == "__main__":
    
    directory_save = '../network_outputs/blender_ResNetUNet50_bce_0.3_w_photo_1.0_w_smooth_0.05_w_geometric_0.001_0.15_perceptual/'
    directory_save = os.path.join(directory_save, "")
    if not os.path.exists(directory_save):
        os.makedirs(directory_save)
        
    set_num = 0
    depth_scale = 10
    
    dataset = '../data/cars_blender_100'
    train_data_set = data_loader.MultiView_dataset_blender(dataset, transform = data_loader.MultiView_dataset_blender.ToTensor(), train = set_num, num_views=3)
    train_dataloader = DataLoader(train_data_set, batch_size = 2, shuffle=True)
    
    
#     depth_network = SegNet(withSkipConnections=True)
#     depth_network = SegNet(withSkipConnections=False)
    #depth_network = ResNetUNet()
    depth_network = ResNetUNet50()

    
    depth_network.load_state_dict(torch.load("./outputs/2020-08-01/04-33-01/ResNetUNet50_bce_0.3_w_photo_1.0_w_smooth_0.05_w_geometric_0.001.ckpt"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if torch.cuda.is_available():
        depth_network.to(device)
        print('using gpu')
    
    depth_network = depth_network.float()
    num_samples = len(train_dataloader.dataset)

    torch.no_grad()
    depth_network.eval()
    for i in range(0, num_samples, 1):

        data_sample = train_data_set[i]
        if torch.cuda.is_available():
            data_sample['masks'] = data_sample['masks'].to(device).float()
            data_sample['views'] = data_sample['views'].to(device).float()
            data_sample['poses'] = data_sample['poses'].to(device).float()
            data_sample['intrinsics'] = data_sample['intrinsics'].to(device).float()
        
        output = depth_network(data_sample['views'][0].unsqueeze(0))
        

        depth = (output[0][0,0] * data_sample['masks'][0,0]).cpu().detach().numpy()
        mask = output[1][0,0].cpu().detach().numpy()
        image = data_sample['views'][0].permute((1, 2, 0)).cpu().detach().numpy()
        print(np.max(mask))
        mask= (mask>0.5) * 1.0

        depth = (depth * (mask)) 

        print(depth_scale * (1 - np.amax(depth)), depth_scale * (1 -  np.amin(depth[depth!=0])))
        
        im_depth = depth.astype('float32')
        
        print(im_depth.mean(), im_depth.var())
        im_depth_tiff = Image.fromarray(im_depth, 'F')
        
        mask = (mask*255).astype(int)
        depth = (depth*255).astype(int)
        image = (image*255).astype('uint8')
        

        
        #################################### SAVING FILES ###########################################
        print("Saving")
        mask_name = directory_save + 'frame_%06d_mask.jpg' % i
        image_name = directory_save +'frame_%06d_image.jpg' % i
        depth_name = directory_save +'frame_%06d_depth.jpg' % i
        depth_tiff_name = directory_save +'frame_%06d_depth.tiff' % i
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_name, image)
        cv2.imwrite(depth_name, depth)
        cv2.imwrite(mask_name, mask)
        im_depth_tiff.save(depth_tiff_name)

