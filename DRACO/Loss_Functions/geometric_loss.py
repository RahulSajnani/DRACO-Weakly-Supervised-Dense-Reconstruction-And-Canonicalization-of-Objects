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

class Geometric_loss(nn.Module):

    def __init__(self):

        super().__init__()


    def forward(self, data, depth_maps):
        '''
        Computes geometric error between point clouds
        '''

        # Extracting view, pose and mask of the reference image
        reference_view = data['views'][:, 0] # B3HW
        reference_pose = data['poses'][:, 0] # B7
        reference_mask = data['masks'][:, 0] # BHW

        # Extract the extrinsics for reference and the target views
        intrinsics = data['intrinsics'] # B33

        # Extract the depth map for reference view
        reference_depth = depth_maps[:, 0]

        # Scale the normalised depth
        # reference_depth = helper_functions.sigmoid_2_depth(reference_depth)

        # Computing the geometric loss
        geometric_loss = 0
        num_views = data['num_views'][0] #B1

        for i in range(1, num_views):

            # Extract the target view, pose, mask, and depth
            target_view = data['views'][:, i] # B3HW
            target_pose = data['poses'][:, i] # B7
            target_mask = data['masks'][:, i] # BHW
            target_depth = depth_maps[:, i]

            # Scale the normalised depth
            # target_depth = helper_functions.sigmoid_2_depth(target_depth)

            batch_size = target_depth.size(0)

            target_coord, proj_target_to_ref = project_depth_point_cloud(depth_maps[:, i].unsqueeze(1), intrinsics, target_pose, reference_pose) #[B,3,H*W]
            ref_coord, proj_ref_to_target = project_depth_point_cloud(reference_depth.unsqueeze(1), intrinsics, reference_pose, target_pose) # [B,3,H*W]

            for j in range(batch_size):
                
                target_coord_mask = target_mask[j].squeeze().reshape(1, -1)
                ref_coord_mask = reference_mask[j].squeeze().reshape(1, -1)

                target_coord_view = target_coord[j][:, target_coord_mask.squeeze() == 1.0]
                ref_coord_view = ref_coord[j][:, ref_coord_mask.squeeze() == 1.0]

                proj_ref_to_target_view = proj_ref_to_target[j][:, ref_coord_mask.squeeze() == 1.0]
                proj_target_to_ref_view = proj_target_to_ref[j][:, target_coord_mask.squeeze() == 1.0] #[3, H*W]
                
                forward_projection_loss, _  = chamfer_distance((ref_coord_view.T).unsqueeze(0), (proj_target_to_ref_view.T).unsqueeze(0))
                
                backward_projection_loss, _ = chamfer_distance((target_coord_view.T).unsqueeze(0), (proj_ref_to_target_view.T).unsqueeze(0))
                #print(backward_projection_loss)
                geometric_loss += (forward_projection_loss + backward_projection_loss)

        loss = geometric_loss / (4 * batch_size * (num_views - 1))

        return loss
