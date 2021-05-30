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
import torchvision
import cv2
import json
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
sys.path.append(os.path.abspath(os.path.join('..')))
import helper_functions
from inv_changed import *





# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model[:9]
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
        }

    def forward(self, x):
        loss = 0
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                if self.layer_name_mapping[name] == "relu2_2":
                    loss = x
        return loss

class Photometric_loss(nn.Module):

    '''
    Photometric loss for self supervision

    '''
    def __init__(self, vgg_model, compute_ssim=True, alpha=0.05):
        super().__init__()

        ## SSIM Parameters
        print("Alpha value = ", alpha)
        self.compute_ssim   = compute_ssim
        self.alpha          = alpha
        self.size = 3
        self.stride = 1
        self.padding = 1
       	self.mu_x_pool   = nn.AvgPool2d(self.size, self.stride, padding = self.padding)
        self.mu_y_pool   = nn.AvgPool2d(self.size, self.stride, padding = self.padding)
        self.sig_x_pool  = nn.AvgPool2d(self.size, self.stride, padding = self.padding)
        self.sig_y_pool  = nn.AvgPool2d(self.size, self.stride, padding = self.padding)
        self.sig_xy_pool = nn.AvgPool2d(self.size, self.stride, padding = self.padding)

        self.refl = nn.ReflectionPad2d(0)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        self.perceptual_loss_init(vgg_model)


    def perceptual_loss_init(self, vgg_model):
        '''
        Initializing perceptual loss network
        '''
        self.perceptual_loss_net = LossNetwork(vgg_model)#.type_as(vgg_model)
        self.perceptual_loss_net.eval()#.type_as(vgg_model)

    def ssim(self, x, y, mask):

        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        temp  = torch.mean(SSIM_n/SSIM_d, axis=1).unsqueeze(1)

        return torch.clamp((1.0 - temp[mask>0.5])/2, 0, 1).mean()

    def forward(self, data, depth, backward_consistency = False):
        '''
        Function to compute loss
        TODO
        - Compute gradient of depth
        '''

        # Scale the normalised depth
        #for i in range(len(depth)):
        #    depth[i] = helper_functions.sigmoid_2_depth(depth[i])

        # Extracting view, pose and mask of the reference image
        reference_view = data['views'][:, 0] # B3HW
        reference_pose = data['poses'][:, 0] # B7
        reference_mask = data['masks'][:, 0] # BHW
        reference_depth = depth[:, 0].float() * data['masks'][:, 0].float()

        # Extracting the region of interest from the reference view by masking out the background
        reference_view = reference_view * reference_mask # B3HW = B3HW * B1HW

        # Extract the extrinsics for reference and the target views
        intrinsics = data['intrinsics'] # B33

        # Compute the SSIM loss if flag is set
        if self.compute_ssim:
            ssim_loss = 0

        perceptual_loss = 0
        # Computing the reconstruction loss
        reconstruction_loss = 0
        num_views = data['num_views'][0] #B1

        if torch.cuda.is_available():
            self.perceptual_loss_net.cuda()

        for i in range(1, num_views):

            # Extract the target view, pose, and mask
            target_view = data['views'][:, i] # B3HW
            target_pose = data['poses'][:, i] # B7
            target_mask = data['masks'][:, i] # BHW
            target_depth = depth[:, i].float() * data["masks"][:, i].float()

            # Extracting the region of interest from the target view by masking out the background
            target_view = target_view * target_mask # B3HW = B3HW * B1HW


            # Warp the reference view to the target views
            warped_view, valid_points = inverse_warp(target_view, reference_depth, intrinsics, reference_pose, target_pose)

            if backward_consistency:
                warped_view_target, valid_points_target = inverse_warp(reference_view, target_depth, intrinsics, target_pose, reference_pose)

            # Only the points within the fov are valid

            #valid_points = valid_points.unsqueeze(1)

            mask_region = reference_mask

            # Compute the intensity difference between the target views and the warped views
            # Only consider the ROI for the difference
            difference = (reference_view - warped_view) * mask_region

            if backward_consistency:
                difference_target = (target_view - warped_view_target) * target_mask

            # Reconstruction loss within mask region
            reconstruction_loss += difference.mean(1).unsqueeze(1)[mask_region > 0.5].abs().mean()

            if backward_consistency:
                reconstruction_loss += difference_target.mean(1).unsqueeze(1)[target_mask > 0.5].abs().mean()

            if self.compute_ssim:
                # SSIM loss within mask region
                ssim_loss += self.ssim(reference_view * mask_region, warped_view * mask_region, mask_region)
                if backward_consistency:

                    ssim_loss += self.ssim(target_view * target_mask, warped_view_target * target_mask, target_mask)

            perceptual_loss +=  (self.perceptual_loss_net((reference_view * mask_region).float()) - self.perceptual_loss_net((warped_view * mask_region).float())).abs().mean()

            if backward_consistency:

                perceptual_loss +=  (self.perceptual_loss_net((target_view * target_mask).float()) - self.perceptual_loss_net((warped_view_target * target_mask).float())).abs().mean()

        # If computing SSIM
        if self.compute_ssim:
            loss = self.alpha * ssim_loss + ( 1 - self.alpha) * reconstruction_loss
        else:
            loss = reconstruction_loss

        loss += perceptual_loss
        #print(perceptual_loss)

        # Compute the average loss per target view
        loss = loss / ((num_views - 1))

        if backward_consistency:
            loss = loss / 2

        return loss


class Photometric_loss_nocs(nn.Module):

    '''
    Photometric loss for self supervision

    '''
    def __init__(self, compute_ssim=True, alpha=0.05):
        super().__init__()

        ## SSIM Parameters
        print("Alpha value = ", alpha)
        self.compute_ssim   = compute_ssim
        self.alpha          = alpha
        self.size = 3
        self.stride = 1
        self.padding = 1
       	self.mu_x_pool   = nn.AvgPool2d(self.size, self.stride, padding = self.padding)
        self.mu_y_pool   = nn.AvgPool2d(self.size, self.stride, padding = self.padding)
        self.sig_x_pool  = nn.AvgPool2d(self.size, self.stride, padding = self.padding)
        self.sig_y_pool  = nn.AvgPool2d(self.size, self.stride, padding = self.padding)
        self.sig_xy_pool = nn.AvgPool2d(self.size, self.stride, padding = self.padding)

        self.refl = nn.ReflectionPad2d(0)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2



    def ssim(self, x, y, mask):

        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        temp  = torch.mean(SSIM_n/SSIM_d, axis=1).unsqueeze(1)

        return torch.clamp((1.0 - temp[mask>0.5])/2, 0, 1).mean()

    def forward(self, data, depth, backward_consistency = False):
        '''
        Function to compute loss

        - Compute gradient of depth
        '''


        # Scale the normalised depth
        #for i in range(len(depth)):
        #   depth[i] = helper_functions.sigmoid_2_depth(depth[i].detach())

        reference_depth = depth[:, 0].float() * data['masks'][:, 0].float()

        # Extracting view, pose and mask of the reference image
        reference_view = data['nocs'][:, 0] # B3HW
        reference_pose = data['poses'][:, 0] # B7
        reference_mask = data['masks'][:, 0] # BHW

        # Extracting the region of interest from the reference view by masking out the background
        reference_view = reference_view * reference_mask # B3HW = B3HW * B1HW

        # Extract the extrinsics for reference and the target views
        intrinsics = data['intrinsics'] # B33

        # Compute the SSIM loss if flag is set
        if self.compute_ssim:
            ssim_loss = 0

        perceptual_loss = 0
        # Computing the reconstruction loss
        reconstruction_loss = 0
        num_views = data['num_views'][0] #B1

#        if torch.cuda.is_available():
 #           self.perceptual_loss_net.cuda()
        for i in range(1, num_views):

            # Extract the target view, pose, and mask
            target_view = data['nocs'][:, i] # B3HW
            target_pose = data['poses'][:, i] # B7
            target_mask = data['masks'][:, i] # BHW
            target_depth = depth[:, i].float() * data["masks"][:, i].float()

            # Extracting the region of interest from the target view by masking out the background
            target_view = target_view * target_mask # B3HW = B3HW * B1HW


            # Warp the reference view to the target views
            warped_view, valid_points = inverse_warp(target_view, reference_depth, intrinsics, reference_pose, target_pose)

            if backward_consistency:
                warped_view_target, valid_points_target = inverse_warp(reference_view, target_depth, intrinsics, target_pose, reference_pose)
            # Only the points within the fov are valid

            mask_region = reference_mask

            # Compute the intensity difference between the target views and the warped views
            # Only consider the ROI for the difference
            difference = (reference_view - warped_view) * mask_region

            if backward_consistency:
                difference_target = (target_view - warped_view_target) * target_mask

            # Reconstruction loss within mask region
            reconstruction_loss += difference.mean(1).unsqueeze(1)[mask_region > 0.5].abs().mean()

            if backward_consistency:
                reconstruction_loss += difference_target.mean(1).unsqueeze(1)[target_mask > 0.5].abs().mean()


            if self.compute_ssim:
                # SSIM loss within mask region
                ssim_loss += self.ssim(reference_view * mask_region, warped_view * mask_region, mask_region)

                if backward_consistency:
                    ssim_loss += self.ssim(target_view * target_mask, warped_view_target * target_mask, target_mask)


        # If computing SSIM
        if self.compute_ssim:
            loss = self.alpha * ssim_loss + ( 1 - self.alpha) * reconstruction_loss
        else:
            loss = reconstruction_loss

        # Compute the average loss per target view
        loss = loss /((num_views - 1))

        if backward_consistency:
            loss = loss / 2

        return loss
