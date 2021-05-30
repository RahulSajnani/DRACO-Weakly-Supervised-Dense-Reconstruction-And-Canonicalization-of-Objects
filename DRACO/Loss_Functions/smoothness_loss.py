import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import glob
import os
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import json
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
sys.path.append(os.path.abspath(os.path.join('..')))
import helper_functions
from inv_changed import *

class Smoothness_loss(nn.Module):

    '''
    Smoothness loss for smoothness of depth map obtained

    '''
    def __init__(self):

        super().__init__()


    def forward(self, depth_maps, data_sample):

        return self.compute_Loss(depth_maps, data_sample)

    def gradient(self, depth_maps):

        D_dx = depth_maps[:, :, :, 1:] - depth_maps[:, :, :, :-1]
        D_dy = depth_maps[:, :, 1:, :] - depth_maps[:, :, :-1, :]
        return D_dx, D_dy

    def compute_Loss(self, depth_maps, data_sample):
        '''
        Function to compute loss

        '''

        # Get the ground truth masks and the ground truth reference view
        mask = data_sample['masks'][:, 0]
        reference_view = data_sample['views'][:, 0] * mask

        # Compute the first order spatial gradients of the reference view in the x and y direction
        d_x_image, d_y_image = self.gradient(reference_view)

        # Compute the weights per pixel for smoothnes term
        d_x_image_weight = torch.exp(-torch.mean(torch.abs(d_x_image), dim=1, keepdim=True))
        d_y_image_weight = torch.exp(-torch.mean(torch.abs(d_y_image), dim=1, keepdim=True))

        ## Convert sigmoid depth to actual depth
        #depth_maps = helper_functions.sigmoid_2_depth(depth_maps)

        # Compute the first order spatial gradients of the depth maps
        d_x, d_y    = self.gradient(depth_maps)

        # Weigh the gradients of the depth maps with the data term weights obtained from the reference view
        d_x = d_x_image_weight * d_x
        d_y = d_y_image_weight * d_y

        # 0-pad in order to maintain shape
        d_x = F.pad(d_x, (0, 1, 0, 0), mode = "replicate")
        d_y = F.pad(d_y, (0, 0, 0, 1), mode = "replicate")

        # Compute the second order spatial gradients of the reference view in the x and y direction
        d_x2, d_xy  = self.gradient(d_x)
        d_yx, d_y2  = self.gradient(d_y)

        # 0-pad in order to maintain shape
        d_x2 = F.pad(d_x2, (0, 1, 0, 0), mode = "replicate")
        d_xy = F.pad(d_xy, (0, 0, 0, 1), mode = "replicate")
        d_yx = F.pad(d_yx, (0, 1, 0, 0), mode = "replicate")
        d_y2 = F.pad(d_y2, (0, 0, 0, 1), mode = "replicate")

        # Compute the smoothness loss as the sum of means of the first and second orders gradients within the masked region
        loss = d_x[mask>0.5].abs().mean() + d_y[mask>0.5].abs().mean()# + d_x2[mask>0.5].abs().mean() + d_xy[mask>0.5].abs().mean() + d_yx[mask>0.5].abs().mean() + d_y2[mask>0.5].abs().mean()

        return loss


class Smoothness_loss_nocs(nn.Module):

    '''
    Smoothness loss for smoothness of depth map obtained

    '''
    def __init__(self):

        super().__init__()


    def forward(self, depth_maps, data_sample):

        return self.compute_Loss(depth_maps, data_sample)

    def gradient(self, depth_maps):

        D_dx = depth_maps[:, :, :, 1:] - depth_maps[:, :, :, :-1]
        D_dy = depth_maps[:, :, 1:, :] - depth_maps[:, :, :-1, :]
        return D_dx, D_dy

    def compute_Loss(self, depth_maps, data_sample):
        '''
        Function to compute loss

        '''

        # Get the ground truth masks and the ground truth reference view
        mask = data_sample['masks'][:, 0]
        reference_view = data_sample['views'][:, 0] * mask

        # Compute the first order spatial gradients of the reference view in the x and y direction
        d_x_image, d_y_image = self.gradient(reference_view)

        # Compute the weights per pixel for smoothnes term
        d_x_image_weight = torch.exp(-torch.mean(torch.abs(d_x_image), dim=1, keepdim=True))
        d_y_image_weight = torch.exp(-torch.mean(torch.abs(d_y_image), dim=1, keepdim=True))

        # Convert sigmoid depth to actual depth
        # depth_maps = helper_functions.sigmoid_2_depth(depth_maps)

        # Compute the first order spatial gradients of the depth maps
        d_x, d_y    = self.gradient(depth_maps)

        # Weigh the gradients of the depth maps with the data term weights obtained from the reference view
        d_x = d_x_image_weight * d_x
        d_y = d_y_image_weight * d_y

        # 0-pad in order to maintain shape
        d_x = F.pad(d_x, (0, 1, 0, 0), mode = "replicate")
        d_y = F.pad(d_y, (0, 0, 0, 1), mode = "replicate")

        # Compute the second order spatial gradients of the reference view in the x and y direction
        d_x2, d_xy  = self.gradient(d_x)
        d_yx, d_y2  = self.gradient(d_y)

        # 0-pad in order to maintain shape
        d_x2 = F.pad(d_x2, (0, 1, 0, 0), mode = "replicate")
        d_xy = F.pad(d_xy, (0, 0, 0, 1), mode = "replicate")
        d_yx = F.pad(d_yx, (0, 1, 0, 0), mode = "replicate")
        d_y2 = F.pad(d_y2, (0, 0, 0, 1), mode = "replicate")

        # Compute the smoothness loss as the sum of means of the first and second orders gradients within the masked region
        loss = (d_x.abs().sum() + d_y.abs().sum())/ (3*torch.sum(mask > 0.5)) # + d_x2[mask>0.5].abs().mean() + d_xy[mask>0.5].abs().mean() + d_yx[mask>0.5].abs().mean() + d_y2[mask>0.5].abs().mean()

        return loss
