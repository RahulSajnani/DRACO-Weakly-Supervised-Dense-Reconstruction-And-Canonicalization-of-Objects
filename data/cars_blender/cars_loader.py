from __future__ import division
import argparse
import numpy as np
from glob import glob
# from joblib import Parallel, delayed
from pebble import ProcessPool
from tqdm import tqdm
import os
import cv2
import json
import imageio

class CarsLoader(object):
    def __init__(self,
                 dataset_dir,
                 sample_gap=1,
                 img_height=480,
                 img_width=640,
                 seq_length=3):
        
        self.dataset_dir = dataset_dir
        self.sample_gap = sample_gap
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.frames = self.get_frames()
        self.num_frames = len(self.frames)

    def get_frames(self):
        img_dir = self.dataset_dir 
        seq_list = os.listdir(img_dir)
        frames = []

        for seq in seq_list:
            
            color00_img_files   = sorted(glob(img_dir + '/' + seq + '/frame_*_Color_00.png'))
            mask00_img_file     = sorted(glob(img_dir + '/' + seq + '/frame_*_Mask_00.png'))
            depth_img_file      = sorted(glob(img_dir + '/' + seq + '/frame_*_Depth_00.exr'))
            extrinsics_file     = sorted(glob(img_dir + '/' + seq + '/frame_*_CameraPose.json'))
            keypoints_file      = sorted(glob(img_dir + '/' + seq + '/frame_*_KeyPoints.npy'))
            
            for f in color00_img_files:
                frame_id = seq + '_' + os.path.basename(f).split('_')[1]
                frames.append(frame_id)
        return frames

    def get_sample_with_idx(self, target_idx):
        target_frame_id = self.frames[target_idx]
        if not self.is_valid_sample(target_frame_id):
            return False
        sample = self.load_sample(target_frame_id)
        return sample

    def load_sample(self, target_frame_id):
        color00_img_seq, mask_img_seq, depth_img_seq, extrinsics_seq, keypoints_seq = self.load_image_sequence(target_frame_id)
        sample = {}
        sample['color00_img_seq']       = color00_img_seq
        sample['mask00_img_seq']        = mask_img_seq
        sample['depth_img_seq']         = depth_img_seq
        sample['extrinsics_seq']        = extrinsics_seq
        sample['keypoints_seq']         = keypoints_seq
        sample['folder_name']           = target_frame_id.split('_')[0]
        sample['file_name']             = target_frame_id.split('_')[-1]
        return sample

    def load_image_sequence(self, target_frame_id):
        seq, local_target_frame_id = target_frame_id.split('_')
        half_offset = int((self.seq_length-1)/2 * self.sample_gap)

        color00_img_seq     = []
        mask00_img_seq      = []
        depth_img_seq       = []
        extrinsics_seq      = []
        keypoints_seq       = []


        for offset in range(-half_offset, half_offset+1, self.sample_gap):
            current_local_frame_id = '%.8d' % (int(local_target_frame_id) + offset)

            current_color00_img_file    = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_Color_00.png")
            current_mask00_img_file     = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_Mask_00.png")
            current_depth_img_file      = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_Depth_00.exr")
            current_extrinsics_file     = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_CameraPose.json")
            current_keypoints_file      = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_KeyPoints.npy")

            current_color00_img     = cv2.imread(current_color00_img_file)
            current_mask00_img      = np.rint(imageio.imread(current_mask00_img_file) / 65535.0) * 255.0 # 0 - 255
            current_depth_img       = imageio.imread(current_depth_img_file)[:, :, 0]
            current_depth_img       = self.decode_depth(current_depth_img) # 0 - 255 range
            current_keypoints_img   = np.load(current_keypoints_file)

            with open(current_extrinsics_file) as f:
                current_extrinsics = f.read()
            
            color00_img_seq.append(current_color00_img)
            mask00_img_seq.append(current_mask00_img)
            keypoints_seq.append(current_keypoints_img)
            depth_img_seq.append(current_depth_img)
            extrinsics_seq.append(current_extrinsics)

        return color00_img_seq, mask00_img_seq, depth_img_seq, extrinsics_seq, keypoints_seq
    

    def is_valid_sample(self, target_frame_id):
        seq, local_target_frame_id = target_frame_id.split('_')
        half_offset = int((self.seq_length-1)/2 * self.sample_gap)
        for offset in range(-half_offset, half_offset+1, self.sample_gap):
            current_local_frame_id = '%.8d' % (int(local_target_frame_id) + offset)

            current_color00_img_file    = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_Color_00.png")
            current_mask00_img_file     = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_Mask_00.png")
            current_depth_img_file      = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_Depth_00.exr")
            current_extrinsics_file     = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_CameraPose.json")
            current_keypoints_file      = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_KeyPoints.npy")

            if not os.path.exists(current_color00_img_file):
                return False
        return True

    def decode_depth(self, depth_image):
        '''
        Obtain depth map from exr file
        '''   
        
        depth_image = (20 - depth_image) / 20 * 255.0
        depth_image[depth_image < 0] = 0
        
        return depth_image

