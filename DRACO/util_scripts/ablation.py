import sys, os
sys.path.append(os.path.abspath(os.path.join('./models')))
sys.path.append(os.path.abspath(os.path.join('./Data_Loaders')))
sys.path.append(os.path.abspath(os.path.join('./Loss_Functions')))

# Importing the lightning module and network
import pytorch_lightning as pl
from trainer import CoolSystem

# Importing the models
from arch import SegNet, SegNet_Split
from arch_2 import Depth_Net
from resnet import ResNetUNet, ResNetUNet50
from densenet_2 import FCDenseNet103

# Import the loss functions for evaluation
import loss_functions, smoothness_loss, geometric_loss, photometric_loss

# Importing the data loader
import data_loader
from torch.utils.data import Dataset, DataLoader

# Importing Utils
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
import torchvision
import torch
import tqdm
import gc
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


BCELoss          = torch.nn.BCELoss()
Photometric_loss = photometric_loss.Photometric_loss(vgg_model=torchvision.models.vgg16(pretrained=True).eval().cuda(), alpha = 0.15)
Smoothness_loss  = smoothness_loss.Smoothness_loss()
Geometric_loss   = geometric_loss.Geometric_loss()

w_photo          = 1.0
w_geometric      = 0.001
w_smooth         = 0.05
w_bce            = 0.3


if __name__ == "__main__":
    
        
    # Path to the input dataset    
    set_num = 3
    dataset = '../../../Self-supervised-NOCS/data/ablation_dataset_wo_depth/'
    
    # Load the checkpoints
    depth_network = ResNetUNet50()
    depth_network.load_state_dict(torch.load('../../../Self-supervised-NOCS/depth-network/outputs/2020-10-26/photo_ssim/ResNetUNet50_bce_1.0_w_photo_1.0_w_smooth_0.15_w_geometric_0.0.ckpt'))
    
    # Load the dataset
    train_data_set = data_loader.MultiView_canonical_dataset_blender(dataset, transform = data_loader.MultiView_dataset_blender.ToTensor(), train = set_num, num_views=3, gt_depth=False, normalize = True, canonical = True)

    train_dataloader = DataLoader(train_data_set, batch_size = 1, shuffle=True)
    
    # Move to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        depth_network = depth_network.to(device)

    
    depth_network = depth_network.float()
    num_samples = len(train_dataloader.dataset)
        
    # Set the grad tracking to false, for evaluation
    torch.no_grad()
    depth_network.eval()
    
    gt_photometric_loss = []
    gt_geometric_loss = []
    gt_smoothness_loss = []
    gt_mask_loss = []
    gt_total_loss = []
    depth_l1_loss = []

    for epoch in range(1):
        depth_network.eval()
        print("%"*20)
        print(f"Epoch: {epoch}")
        print("%"*20)
        
        for batch, data_sample in enumerate(train_dataloader):

            if torch.cuda.is_available():
                
                #data_sample['depths'] = data_sample['depths'].to(device).float()
                data_sample['intrinsics'] = data_sample['intrinsics'].to(device).float()
                data_sample['masks'] = data_sample['masks'].to(device).float()
                data_sample['views'] = data_sample['views'].to(device).float()
                data_sample['poses'] = data_sample['poses'].to(device).float()

            gt_mask_ref = data_sample['masks'][:, 0]
            #gt_depth_ref = data_sample['depths'].to(device).float()
            #gt_depth_ref[gt_depth_ref == np.inf] = 10.0
            #gt_depth_ref = 1.0 - (gt_depth_ref / 10.0)
                
                
            
            # Forward Pass
            output = depth_network(data_sample['views'][:, 0])
            pr_target_depth = [output[0].to(device).float()]
            for i in range(1, data_sample['num_views'][0]):
                pr_target_depth.append(depth_network(data_sample['views'][:, i])[0].to(device).float())
            pr_target_depth = torch.cat(pr_target_depth, dim=1)
                    
                
            # Compute Ground Truth Loss
            gt_photometric_loss.append( Photometric_loss(data_sample, pr_target_depth[:, 0]).data)
            gt_geometric_loss.append(   Geometric_loss(data_sample, pr_target_depth.squeeze(2)).data)
            #gt_smoothness_loss.append(  Smoothness_loss(pr_target_depth[:, 0], data_sample).data)
            gt_mask_loss.append(        BCELoss(gt_mask_ref[:, 0], gt_mask_ref[:, 0]).data)
            #depth_l1_loss.append(       torch.mean(torch.abs(gt_depth_ref[:, 0] - output[0])))
           # gt_total_loss.append(       w_photo * gt_photometric_loss[-1]   + w_smooth * gt_smoothness_loss[-1] + w_bce * gt_mask_loss[-1] + w_geometric * gt_geometric_loss[-1])
            
            #if batch >= 4:
            #    break
            # print("gt_photometric_loss:", gt_photometric_loss[-1])
            # print("gt_geometric_loss:", gt_geometric_loss[-1])
            # print("gt_smoothness_loss:", gt_smoothness_loss[-1])
            # print("gt_mask_loss:", gt_mask_loss[-1])
            # print("gt_total_loss:", gt_total_loss[-1])
            # print()
            
            #torch.cuda.empty_cache()
            #gc.collect()

        # GPUtil.showUtilization()

        print(torch.mean(torch.tensor(gt_photometric_loss)))
        print(torch.mean(torch.tensor(gt_geometric_loss)))
        #print(torch.mean(torch.tensor(gt_smoothness_loss)))
        #print(torch.mean(torch.tensor(gt_mask_loss)))
        #print(torch.mean(torch.tensor(depth_l1_loss)))
        
