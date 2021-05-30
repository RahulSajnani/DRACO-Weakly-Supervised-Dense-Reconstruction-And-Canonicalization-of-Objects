import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import glob
import os, sys
import torch
#import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import json
sys.path.append(os.path.abspath(os.path.join('../')))
import helper_functions
import inv_changed
import torchvision
import imageio




class MultiView_canonical_dataset_blender(Dataset):

    '''
    XNOCS multiview data loader
    '''

    def __init__(self, dataset_path, transform = None, train = 1, num_views = 3, gt_depth = True, gt_nocs = False, normalize = True, jitter = False, canonical = False):
        
        '''
        init class method
        '''

        self.canonical = canonical
        self.normalize = normalize
        self.jitter = jitter
        print('Retrieving data')
        self.gt_depth = gt_depth
        self.gt_nocs  = gt_nocs
        self.K = self.construct_camera_matrix(888.88, 1000, 320, 240)
        self.transform = transform
        self.num_views = num_views
        self.Views, self.Masks, self.Depth, self.NOCS, self.Pose, self.keypoints, self.cam_coords, self.rotation = self.load_image_names(dataset_path, train)

        self.color_jitter_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ColorJitter(brightness=0.35, contrast=0.5, saturation=0.5, hue = 0.5),       
        ])   
        # print(self.Views[:10], '\n', self.Masks[:10], '\n', self.Pose[:10])
        print(len(self.Views), len(self.Masks), len(self.Pose), len(self.NOCS))
        print('Data retrieved')

    def __len__(self):

        '''
        length of dataset
        '''
        return len(self.Views)

    
        
    def construct_camera_matrix(self, focal_x, focal_y, c_x, c_y):
        '''
        Obtain camera intrinsic matrix
        '''

        K = np.array([[focal_x,       0,     c_x],
                    [        0, focal_y,     c_y],
                    [        0,       0,       1]])

        return K

    def __getitem__(self, index):

        '''
        Retrieve data item
        '''

        
        image_rgb_concat = cv2.imread(self.Views[index]) 
        image_rgb_concat = cv2.cvtColor(image_rgb_concat, cv2.COLOR_BGR2RGB) 

        if self.jitter:
            image_rgb_concat = np.array(self.color_jitter_transform(image_rgb_concat))
            # print(image_rgb_concat.shape)
        
        image_rgb_concat = image_rgb_concat / 255.0
        # print("max value:", image_rgb_concat.max())
        
        image_mask_concat = cv2.imread(self.Masks[index], cv2.IMREAD_GRAYSCALE) / 255.0
        image_mask_concat = image_mask_concat.reshape(image_mask_concat.shape[0], image_mask_concat.shape[1], 1)
        image_mask_concat = np.rint(image_mask_concat)

        if self.canonical:
            keypoints_concat = np.load(self.keypoints[index])
            cam_coords_concat = np.load(self.cam_coords[index])[:, 0]
            rotation_concat = np.load(self.rotation[index])[:, 0]
            # print(rotation_concat.shape, cam_coords_concat.shape)
        else:
            keypoints_concat = None
            cam_coords_concat = None
            rotation_concat = None

        if self.gt_depth:
            image_depth_concat = imageio.imread(self.Depth[index])
            image_depth_concat = image_depth_concat.reshape(image_depth_concat.shape[0], image_depth_concat.shape[1], 1)     
        else:
            image_depth_concat = None

        if self.gt_nocs:
            image_nocs_concat = cv2.imread(self.NOCS[index]) 
            image_nocs_concat = cv2.cvtColor(image_nocs_concat, cv2.COLOR_BGR2RGB) 
            # image_nocs_concat = imageio.imread(self.NOCS[index])
            image_nocs_concat = image_nocs_concat / 255.0
        else:
            image_nocs_concat = None
            
        camera_pose = helper_functions.read_json_file(self.Pose[index])
        
        images_dictionary = self.split_image(image_rgb_concat, image_mask_concat, camera_pose, self.num_views, view_depth=image_depth_concat, view_nocs=image_nocs_concat, keypoints_param=keypoints_concat, cam_coords = cam_coords_concat, rotation = rotation_concat)
        images_dictionary['intrinsics'] = self.K
        
        # if self.transform:
        transform = self.ToTensor()
        data = transform(images_dictionary, num_views = self.num_views, gt_depth = self.gt_depth, gt_nocs = self.gt_nocs, normalize = self.normalize, canonical = self.canonical)
        
        
        return data
    
    def load_image_names(self, path, train = 1):
        '''
        Load names of image, mask, and camera pose file names
        '''
        if train == 1:
            data_file_name = 'train.txt'
        elif train == 2:
            data_file_name = 'overfit.txt'
        else:
            data_file_name = 'val.txt'

        file_path = os.path.join(path, data_file_name)
        
        fp = open(file_path, 'r')

        camera_pose = []
        mask = []
        view = []
        depth = []
        nocs = []

        keypoints = []

        c3dpo_cam_coords = []
        c3dpo_rotation = []

        for line in fp:
            fields = line.split(' ')
            
            if 'view' in fields[1]:
                
                name = fields[1] + '.jpg'
                view_number = fields[1].split('_')[1].split('\n')[0]
                
                view_name = os.path.join(path, fields[0], 'view_' + view_number + '.jpg')
                mask_name = os.path.join(path, fields[0], 'mask_' + view_number + '.jpg')
                
                if self.gt_depth:
                    depth_name = os.path.join(path, fields[0], 'depth_' + view_number + '.tiff')
                    depth.append(depth_name)

                if self.gt_nocs:
                    nocs_name = os.path.join(path, fields[0], 'nocs_' + view_number + '.jpg')
                    nocs.append(nocs_name)
                
                pose_name = os.path.join(path, fields[0],('CameraPose_' + view_number + '.json'))
                
                if self.canonical:
                    keypoint_name = os.path.join(path, fields[0], "keypoints_" + view_number + ".npy")        
                    keypoints.append(keypoint_name)
                
                    c3dpo_cam_coords_name = os.path.join(path, fields[0], "c3dpo_" + view_number + "_cam_coords.npy")        
                    c3dpo_cam_coords.append(c3dpo_cam_coords_name)
                
                    c3dpo_rotation_name = os.path.join(path, fields[0], "c3dpo_" + view_number + "_rotation.npy")        
                    c3dpo_rotation.append(c3dpo_rotation_name)
                
                
                view.append(view_name)
                mask.append(mask_name)
                camera_pose.append(pose_name)
        
        view = sorted(view)
        mask = sorted(mask)
        camera_pose = sorted(camera_pose)
        
        if self.gt_depth:
            depth = sorted(depth)
        else:
            depth = None

        if self.gt_nocs:
            nocs = sorted(nocs)
        else:
            nocs = None

        if self.canonical:
            keypoints = sorted(keypoints)
            c3dpo_cam_coords = sorted(c3dpo_cam_coords)
            c3dpo_rotation = sorted(c3dpo_rotation)
        else:
            keypoints = None
            c3dpo_cam_coords = None
            c3dpo_rotation = None

        return view, mask, depth, nocs, camera_pose, keypoints, c3dpo_cam_coords, c3dpo_rotation

    def split_image(self, view_image, view_mask, pose_params, num_views = 3, view_depth = None, view_nocs = None, keypoints_param = None, cam_coords = None, rotation = None):

        '''
        Split the images in respective views
        Arguments:
            view_image   :   H x W x 3 - image of concatenated views
            num_views    :      scalar - number of views to split image
        
        Returns:
            images_split :  dictionary - index and image
        '''

        view_list = []
        mask_list = []
        pose_list = []
        depth_list = []
        nocs_list = []

        keypoints_list = []
        rotations_list = []
        cam_coords_list = []

        width = int(view_image.shape[1] / num_views)
        images_split = {}
        images_split['num_views'] = num_views

        for i in range(num_views):

            index = i - int(num_views / 2)

            
            # Extracting parameters

            pose = pose_params[i]
            pose = self.pose_dict_to_numpy(pose)

            image = view_image[ :, i*width:(i + 1)*width].transpose((2, 0, 1))
            mask = view_mask[ :, i*width:(i + 1)*width].transpose((2, 0, 1))
            
            if self.canonical:

                keypoint = keypoints_param[i]
                cam_coord_cur = cam_coords[i]
                rotation_cur = rotation[i]
            # Inserting in list to stack
            if index == 0:
                view_list.insert(index, image)
                mask_list.insert(index, mask)
                pose_list.insert(index, pose)

                if self.canonical:
                    keypoints_list.insert(index, keypoint)
                    cam_coords_list.insert(index, cam_coord_cur)
                    rotations_list.insert(index, rotation_cur)        
            
            else:
                view_list.append(image)
                mask_list.append(mask)
                pose_list.append(pose)
                
                if self.canonical:
                    keypoints_list.append(keypoint)
                    cam_coords_list.append(cam_coord_cur)
                    rotations_list.append(rotation_cur)
                
            
            if view_depth is not None:
                depth = view_depth[ :, i*width:(i + 1)*width].transpose((2, 0, 1))
                if index == 0:
                    depth_list.insert(index, depth)
                else:
                    depth_list.append(depth)


            if view_nocs is not None:
                nocs = view_nocs[ :, i*width:(i + 1)*width].transpose((2, 0, 1))
                if index == 0:
                    nocs_list.insert(index, nocs)
                else:
                    nocs_list.append(nocs)

            

        images_split["views"] = np.stack(view_list)
        images_split["poses"] = np.stack(pose_list)
        images_split["masks"] = np.stack(mask_list)

        if self.gt_depth:
            images_split["depths"] = np.stack(depth_list)

        if self.gt_nocs:
            images_split["gt_nocs"] = np.stack(nocs_list)
        
        if self.canonical:
            images_split["keypoints"] = np.stack(keypoints_list)
            images_split["c3dpo_rotation"] = np.stack(rotations_list)
            images_split["c3dpo_cam_coords"] = np.stack(cam_coords_list)
        
        return images_split

    def pose_dict_to_numpy(self, pose):
        '''
        Convert pose dictionary to numpy array 
        '''
        pose = np.array([pose['position']['x'], 
                         pose['position']['y'], 
                         pose['position']['z'],
                         pose['rotation']['x'], 
                         pose['rotation']['y'], 
                         pose['rotation']['z'], 
                         pose['rotation']['w'] 
                         ])
        return pose

    class ToTensor(object):

        '''
        Convert data to tensor
        '''
        def pose_2_matrix(self, pose):
            '''
            Function to convert pose to transformation matrix
            '''
            flip_x = torch.eye(4)
            flip_x[2, 2] *= -1
            flip_x[1, 1] *= -1
            
            views = pose.size(0)

            rot_mat = inv_changed.quat2mat(pose[:, 3:]) # num_views 3 3
            translation_mat = pose[:, :3].unsqueeze(-1) # num_views 3 1

            transformation_mat = torch.cat([rot_mat, translation_mat], dim = 2)
            transformation_mat = torch.cat([transformation_mat, torch.tensor([[0,0,0,1]]).unsqueeze(0).expand(1,1,4).type_as(transformation_mat).repeat(views, 1, 1)], dim=1)

            flip_x = flip_x.inverse().type_as(transformation_mat)
            
            # 180 degree rotation around x axis due to blender's coordinate system
            return  transformation_mat @ flip_x

        def __call__(self, data, num_views, gt_depth, gt_nocs, normalize, canonical):
            data_trans = {}
            if gt_depth:
                data_trans['depths'] = torch.from_numpy(data["depths"])

            if gt_nocs:
                data_trans['gt_nocs'] = torch.from_numpy(data["gt_nocs"])
            
            if canonical:
                data_trans["keypoints"] = torch.from_numpy(data["keypoints"])
                data_trans["c3dpo_rotation"] = torch.from_numpy(data["c3dpo_rotation"])
                data_trans["c3dpo_cam_coords"] = torch.from_numpy(data["c3dpo_cam_coords"])

            #print(data["poses"].shape)
            data_trans['views'] = torch.from_numpy(data["views"])
            data_trans['masks'] = torch.from_numpy(data["masks"])
            data_trans['poses'] = self.pose_2_matrix(torch.from_numpy(data["poses"])).squeeze() # [4, 4]
            data_trans['intrinsics'] = torch.from_numpy(data['intrinsics'])
            data_trans['num_views'] = torch.tensor(data['num_views'])

            if normalize:
                # https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
                data_trans["views_not_normalized"] = data_trans["views"].clone()
                normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225],
                                                            inplace=True)
                
                
                for view_num in range(data_trans['num_views']):
                    
                    data_trans['views'][view_num] = normalize_transform(data_trans['views'][view_num])

            
            return data_trans

if __name__ == "__main__":


    # dataset_path = "../../data/cars_blender_prepared/"
    dataset_path = "../../../prepare/"
    
    data_set = MultiView_canonical_dataset_blender(dataset_path, train = 1,  normalize=True, jitter=True, canonical=True)
    # data_set = MultiView_dataset_blender(dataset_path, train = 1, gt_depth = False, normalize=True, jitter=True, transform = MultiView_dataset_blender.ToTensor())

    
    data = data_set[0]["depths"]
    data = data_set[0]["gt_nocs"]
    print(torch.unique(data))
    # print(data.var(), data.max())
    data_loader = DataLoader(data_set, batch_size = 2, shuffle=True)
    
    print(data_set[0].keys())
    for batch, data_sample in enumerate(data_loader):
        
        for key in data_sample:

            print(key, " ", data_sample[key].shape)

        print("###########################################")
