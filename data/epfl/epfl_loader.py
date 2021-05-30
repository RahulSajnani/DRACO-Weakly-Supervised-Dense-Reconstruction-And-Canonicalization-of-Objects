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

class EPFLLoader(object):
    def __init__(self,
                 dataset_dir,
                 sample_gap=1,
                 img_height=250,
                 img_width=376,
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
            
            color00_img_files   = sorted(glob(img_dir + '/' + seq + f'/tripod_{seq}_*.jpg'))
            
            for f in color00_img_files:
                frame_id = seq + '_' + os.path.basename(f).split('_')[3].split('.')[0]
                frames.append(frame_id)
        return frames

    def get_sample_with_idx(self, target_idx):
        target_frame_id = self.frames[target_idx]
        if not self.is_valid_sample(target_frame_id):
            return False
        sample = self.load_sample(target_frame_id)
        return sample

    def load_sample(self, target_frame_id):
        color00_img_seq = self.load_image_sequence(target_frame_id)
        sample = {}
        sample['color00_img_seq']       = color00_img_seq
        sample['folder_name']           = f"seq_{target_frame_id.split('_')[1]}"
        sample['file_name']             = f"{target_frame_id.split('_')[-1]}"
        return sample

    def load_image_sequence(self, target_frame_id):
        _, seq, local_target_frame_id = target_frame_id.split('_')
        half_offset = int((self.seq_length-1)/2 * self.sample_gap)

        color00_img_seq     = []


        for offset in range(-half_offset, half_offset+1, self.sample_gap):
            current_local_frame_id = '%.3d' % (int(local_target_frame_id) + offset)

            current_color00_img_file    = os.path.join(self.dataset_dir, f"seq_{seq}/tripod_seq_{seq}_{current_local_frame_id}.jpg")

            current_color00_img     = cv2.imread(current_color00_img_file)
            
            color00_img_seq.append(current_color00_img)

        return color00_img_seq
    

    def is_valid_sample(self, target_frame_id):
        _, seq, local_target_frame_id = target_frame_id.split('_')
        half_offset = int((self.seq_length-1)/2 * self.sample_gap)
        for offset in range(-half_offset, half_offset+1, self.sample_gap):
            current_local_frame_id = '%.3d' % (int(local_target_frame_id) + offset)

            current_color00_img_file    = os.path.join(self.dataset_dir, f"seq_{seq}/tripod_seq_{seq}_{current_local_frame_id}.jpg")

            if not os.path.exists(current_color00_img_file):
                return False
        return True

