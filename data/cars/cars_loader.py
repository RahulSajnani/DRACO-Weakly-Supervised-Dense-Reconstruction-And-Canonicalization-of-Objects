from __future__ import division
import json
import os
import numpy as np
from glob import glob
import cv2 
import json

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
            color01_img_files   = sorted(glob(img_dir + '/' + seq + '/frame_*_Color_01.png'))
            nox_img_files       = sorted(glob(img_dir + '/' + seq + '/frame_*_NOXRayTL_00.png'))
            depth_img_files     = sorted(glob(img_dir + '/' + seq + '/frame_*_Depth_00.png'))
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
        color00_img_seq, color01_img_seq, nox_img_seq, depth_img_seq, extrinsics_seq, opp_extrinsics_seq = self.load_image_sequence(target_frame_id)
        sample = {}
        sample['color00_img_seq']       = color00_img_seq
        sample['color01_img_seq']       = color01_img_seq
        sample['nox_img_seq']           = nox_img_seq
        sample['depth_img_seq']         = depth_img_seq
        sample['extrinsics_seq']        = extrinsics_seq
        sample['folder_name']           = target_frame_id.split('_')[0]
        sample['file_name']             = target_frame_id.split('_')[-1]
        sample['opp_file_name']         = '%.8d' % (int(target_frame_id.split('_')[-1]) + self.num_frames)
        sample['opp_extrinsics_seq']    = opp_extrinsics_seq
        return sample

    def load_image_sequence(self, target_frame_id):
        seq, local_target_frame_id = target_frame_id.split('_')
        half_offset = int((self.seq_length-1)/2 * self.sample_gap)

        color00_img_seq     = []
        color01_img_seq     = []
        nox_img_seq         = []
        depth_img_seq       = []
        extrinsics_seq      = []
        opp_extrinsics_seq  = []

        for offset in range(-half_offset, half_offset+1, self.sample_gap):
            current_local_frame_id = '%.8d' % (int(local_target_frame_id) + offset)

            current_color00_img_file    = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_Color_00.png")
            current_color01_img_file    = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_Color_01.png")
            current_nox_img_file        = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_NOXRayTL_00.png")
            current_depth_img_file      = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_Depth_00.png")
            current_extrinsics_file     = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_CameraPose.json")

            current_color00_img = cv2.imread(current_color00_img_file)
            current_color01_img = cv2.imread(current_color01_img_file)
            current_nox_img     = cv2.imread(current_nox_img_file)
            current_depth_img   = cv2.imread(current_depth_img_file)
            current_depth_img   = self.decode_depth(current_depth_img)

            with open(current_extrinsics_file) as f:
                current_extrinsics = f.read()

            opp_current_extrinsics = json.loads(current_extrinsics)
            opp_current_extrinsics["position"]["x"] *= -1
            opp_current_extrinsics["position"]["y"] *= -1
            opp_current_extrinsics["position"]["z"] *= -1
            opp_current_extrinsics["rotation"]["w"] *= -1
            
            opp_current_extrinsics = json.dumps(opp_current_extrinsics) +'\n'

            color00_img_seq.append(current_color00_img)
            color01_img_seq.append(current_color01_img)
            nox_img_seq.append(current_nox_img)
            depth_img_seq.append(current_depth_img)
            extrinsics_seq.append(current_extrinsics)
            opp_extrinsics_seq.append(opp_current_extrinsics)
        return color00_img_seq, color01_img_seq, nox_img_seq, depth_img_seq, extrinsics_seq, opp_extrinsics_seq
    

    def is_valid_sample(self, target_frame_id):
        seq, local_target_frame_id = target_frame_id.split('_')
        half_offset = int((self.seq_length-1)/2 * self.sample_gap)
        for offset in range(-half_offset, half_offset+1, self.sample_gap):
            current_local_frame_id = '%.8d' % (int(local_target_frame_id) + offset)

            current_color00_img_file    = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_Color_00.png")
            current_color01_img_file    = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_Color_01.png")
            current_nox_img_file        = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_NOXRayTL_00.png")
            current_depth_img_file      = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_Depth_00.png")
            current_extrinsics_file     = os.path.join(self.dataset_dir, seq, f"frame_{current_local_frame_id}_CameraPose.json")

            if not os.path.exists(current_color00_img_file):
                return False
        return True

    def decode_depth(self, depth_image):
        depth_image = np.array(depth_image)
        # first 16 bits (first 2 channels) are 16-bit depth
        # R is the 8 LSB and G are the others
        depth_image_16 = depth_image[:,:,[1, 0]]
        # B are 8-bit version
        depth_image_8 = depth_image[:,:,2]
        # last 8 are empty
        depth_single_channel = np.zeros((depth_image_16.shape[0], depth_image_16.shape[1]))
        # convert 16 bit to actual depth values
        for i in range(depth_single_channel.shape[0]):
            for j in range(depth_single_channel.shape[1]):
                bit_str = '{0:08b}'.format(depth_image_16[i, j, 0]) + '{0:08b}'.format(depth_image_16[i, j, 1])
                depth_single_channel[i, j] = (8 - int(bit_str, 2)/1000)/8 * 255
                if depth_single_channel[i, j] < 0:
                    depth_single_channel[i, j] = 0

        return depth_single_channel