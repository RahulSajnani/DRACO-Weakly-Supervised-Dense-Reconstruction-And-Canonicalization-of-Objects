
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
from multiprocessing import Pool


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
    
def generate_NOCS(depth_map, json_pose, K, translation_2_center):
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

    point_cloud =  (rotation @ flip_x @ point_cloud + translation[:, np.newaxis] - translation_2_center[:, np.newaxis]) + 0.5
    
    point_cloud = point_cloud.T
    nocs_map = point_cloud.reshape(depth_map.shape[0], depth_map.shape[1], 3)
    nocs_map[depth_map == np.inf] = 1.0
    nocs_map = nocs_map.clip(min=0, max = 1.0)
    
    return nocs_map

def get_range_point_cloud(depth_image, camera_json, K):

    depth_vector = np.reshape(depth_image.copy(), (1, -1))
    mask = (depth_vector != np.inf)*1
    mask = mask[0]
    rotation, translation = get_transformation_matrix(camera_json)
    coords, image_points = get_grid(depth_image.shape[0], depth_image.shape[1])
    
    flip_x = np.eye(3)
    flip_x[1, 1] *= -1
    flip_x[2, 2] *= -1
    
    point_cloud = depth_vector[:, mask == 1] * (np.linalg.inv(K) @ coords[:, mask == 1])
    point_cloud = (rotation @ flip_x @ point_cloud + translation[:, np.newaxis]) 

    point_cloud_min = np.min(point_cloud, axis = 1)
    point_cloud_max = np.max(point_cloud, axis = 1)

    return {"min": point_cloud_min, "max": point_cloud_max}

def generate_nocs_folder(folder):
    '''
    Generates NOCS maps for all files in folder
    '''

    
    f = folder
    image_files = glob.glob(f + '/frame_*_Color_00.png')
    K = construct_camera_matrix(888.88, 1000, 320, 240)
    
    first = True

    for image_path in image_files:

        image_number = image_path.split("_")[-3]
        depth_name = "frame_%08d_Depth_00.exr" % int(image_number)
        camera_pose_name = "frame_%08d_CameraPose.json" % int(image_number)
        depth_path = os.path.join(f, depth_name)
        camera_pose_path = os.path.join(f, camera_pose_name)
        depth_image = imageio.imread(depth_path)[:, :, 0]    
        camera_json = read_json_file(camera_pose_path)

        point_cloud_dict = get_range_point_cloud(depth_image, camera_json, K)
        
        # print("Debug:", point_cloud_dict)
        if first:
            first = False
            point_running_dict = point_cloud_dict
        else:
            for i in range(point_cloud_dict["min"].shape[0]):
                if point_cloud_dict["min"][i] < point_running_dict["min"][i]:
                    point_running_dict["min"][i] = point_cloud_dict["min"][i]    
                    # print("min", i)            
                if point_cloud_dict["max"][i] > point_running_dict["max"][i]:
                    point_running_dict["max"][i] = point_cloud_dict["max"][i]
                    # print("max", i)
    print("Debug: ", point_running_dict)
    translation_2_center = (point_running_dict["min"] + point_running_dict["max"]) / 2
    print("Debug: ", translation_2_center)
    
    for image_path in image_files:
        
        image_number = image_path.split("_")[-3]
        depth_name = "frame_%08d_Depth_00.exr" % int(image_number)
        camera_pose_name = "frame_%08d_CameraPose.json" % int(image_number)
        depth_path = os.path.join(f, depth_name)
        camera_pose_path = os.path.join(f, camera_pose_name)
        depth_image = imageio.imread(depth_path)[:, :, 0]    
        camera_json = read_json_file(camera_pose_path)
        
        nocs_map = generate_NOCS(depth_image, camera_json, K, translation_2_center)
        nocs_file_name = os.path.join(f, "frame_%08d_NOXRayTL_00.png" % int(image_number))
        plt.imsave(nocs_file_name, nocs_map)
    
    
    
def main(args):

    '''
    Script to generate NOCS maps
    '''

    
    path = args.dataset
    folders = glob.glob(path + '/*')
    
    with Pool(args.threads) as p:
        r = list(tqdm.tqdm(p.imap(generate_nocs_folder, folders), total = len(folders)))
        # list(tqdm.tqdm(p.imap(_foo, range(30)), total=30))


 

if __name__=="__main__":

    ################################# Argument Parser #####################################

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help = "Path to parent directory of images dataset", required=True)
    parser.add_argument("--threads", help = "Number of threads", required=False, type=int, default=2)
    
    # parser.add_argument("--model", help = "Model weights", required=True)
    # parser.add_argument("--output", help = "Model weights", required=True)

    args = parser.parse_args()

    main(args)