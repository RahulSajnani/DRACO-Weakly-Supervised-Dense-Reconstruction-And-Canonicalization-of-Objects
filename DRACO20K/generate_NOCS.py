
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import numpy as np
import json
import imageio
import matplotlib.pyplot as plt
import mathutils
import tqdm

'''
Author: Rahul Sajnani

Script to generate NOCS maps from blender generated dataset.
'''


def read_json_file(path):
    '''
    Read json file
    '''
    with open(path) as fp:
        json_data = json.load(fp)

    return json_data

def get_transformation_matrix(pose):
    '''
    Obtain transformation matrix from json pose
    '''

    q = pose["rotation"]
    rotation = mathutils.Quaternion([q["w"], q["x"], q["y"], q["z"]])
    rotation = np.array(rotation.to_matrix())
    
    t = pose["position"]
    translation = np.array([t["x"], t["y"], t["z"]])

    return rotation, translation

def construct_camera_matrix(focal_x, focal_y, c_x, c_y):
    '''
    Obtain camera intrinsic matrix
    '''

    K = np.array([[focal_x,       0,     c_x],
                [        0, focal_y,     c_y],
                [        0,       0,       1]])

    return K

def get_grid(x, y):
    '''
    Get index grid from image
    '''
    
    y_i, x_i = np.indices((x, y))
    image_points = np.stack([x_i, y_i, np.ones_like(x_i)], axis = -1)
    coords = image_points.reshape(x*y, 3)
    
    return coords.T, image_points
    
def generate_NOCS(depth_map, json_pose, K):
    '''
    Function to generate NOCS given depth map and json pose
    '''

    depth_vector = np.reshape(depth_map.copy(), (1, -1))
    depth_vector[depth_vector == np.inf] = 5000
    rotation, translation = get_transformation_matrix(json_pose)
    coords, image_points = get_grid(depth_map.shape[0], depth_map.shape[1])


    flip_x = np.eye(3)
    flip_x[1, 1] *= -1
    flip_x[2, 2] *= -1
    
    point_cloud = depth_vector * (np.linalg.inv(K) @ coords)

    point_cloud =  (rotation @ flip_x @ point_cloud + translation[:, np.newaxis]) + 0.5
    
    point_cloud = point_cloud.T
    nocs_map = point_cloud.reshape(depth_map.shape[0], depth_map.shape[1], 3)
    nocs_map[depth_map == np.inf] = 1.0
    nocs_map = nocs_map.clip(min=0, max = 1.0)
    
    return nocs_map

def main(args):

    '''
    Script to generate NOCS maps
    '''

    
    path = args.dataset
    folders = glob.glob(path + '/*')
    K = construct_camera_matrix(888.88, 1000, 320, 240)
    progress_bar = tqdm.tqdm(range(len(folders)), unit="folder")
    
    for f in folders:
        image_files = glob.glob(f + '/frame_*_Color_00.png')
        
        for image_path in image_files:
            
            image_number = image_path.split("_")[-3]
            
            depth_name = "frame_%08d_Depth_00.exr" % int(image_number)
            camera_pose_name = "frame_%08d_CameraPose.json" % int(image_number)
            depth_path = os.path.join(f, depth_name)
            camera_pose_path = os.path.join(f, camera_pose_name)

            depth_image = imageio.imread(depth_path)[:, :, 0]    
            camera_json = read_json_file(camera_pose_path)
            

            nocs_map = generate_NOCS(depth_image, camera_json, K)
            nocs_file_name = os.path.join(f, "frame_%08d_NOXRayTL_00.png" % int(image_number))
            plt.imsave(nocs_file_name, nocs_map)
            
            # print(nocs_map.max(), nocs_map.min())
        progress_bar.update(1)

    progress_bar.close()

if __name__=="__main__":

    ################################# Argument Parser #####################################

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help = "Path to parent directory of images dataset", required=True)
    
    # parser.add_argument("--model", help = "Model weights", required=True)
    # parser.add_argument("--output", help = "Model weights", required=True)

    args = parser.parse_args()

    main(args)