"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
#import Data_Loaders.data_loader as dl
import glob
#from dataset.dataset_configs import STICKS
from experiment import init_model_from_dir
#from tools.model_io import download_model
from tools.utils import get_net_input

from tools.vis_utils import show_projections
#from visuals.rotating_shape_video import rotating_3d_video
import numpy as np
import json
from tools.so3 import so3_exponential_map

def run_demo(args):

    model, _ = init_model_from_dir(args.model)
    model.eval().cuda()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for name in glob(args.dataset):
        folder = glob(name + '/*')

        output_folder = args.output + '/' + folder.split('/')[-1]
        os.makedirs(output_folder)

        for f in folder:

            output_sub_dir = output_folder + '/' + f.split('/')[-1]
            os.makedirs(output_sub_dir)

            keypoint_files = glob(f + '/*.npy')
            for f in keypoint_files:
                kp = torch.from_numpy(np.load(f)).unsqueeze(0)
                kp = kp.transpose(1,2)
                net_input = {'kp_loc': kp[:, :2, :].contiguous().cuda().float(), "kp_vis": kp[:, 2, :].cuda().float()}
                preds = model(**net_input)

                rot = preds["phi"]["R"].detach().cpu().numpy()
                translation = preds["phi"]["T"].detach().cpu().numpy()
                camera_coordinates = preds["phi"]["shape_camera_coord"].detach().cpu().numpy()
                scale = preds["phi"]["scale"].detach().cpu().numpy()

                file_rot_name = output_sub_dir + '/' + "frame_%08d_rotation.npy" % i
                file_trans_name = output_sub_dir + '/' + "frame_%08d_translation.npy" % i
                file_camera_coords_name = output_sub_dir + '/' + "frame_%08d_cam_coords.npy" % i

                np.save(file_rot_name, rot)
                np.save(file_trans_name, translation)
                np.save(file_camera_coords_name, camera_coordinates)

if __name__=="__main__":

    ################################# Argument Parser #####################################

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help = "Path to images", required=True)
    parser.add_argument("--model", help = "Model weights", required=True)
    parser.add_argument("--output", help = "Model weights", required=True)

    args = parser.parse_args()

    run_demo(args, args.path, args.model)
