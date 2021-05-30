import open3d as o3d
from open3d import *

import numpy as np
import sys, os
import matplotlib.pyplot as plt
import cv2
import torch
import glob
import copy
import mathutils
from PIL import Image
from pytorch3d.loss import chamfer_distance
from tk3dv.nocstools.aligning import estimateSimilarityUmeyama
import math
from tqdm import tqdm
from colorama import Fore, Back, Style
from matplotlib.gridspec import GridSpec


import json

sys.path.append('../')
sys.path.append(os.path.abspath(os.path.join('../models')))
sys.path.append(os.path.abspath(os.path.join('../Data_Loaders')))
sys.path.append(os.path.abspath(os.path.join('../Loss_Functions')))
from aadil_test import *

def quat2mat(x,y,z,w): 

    B = 0
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    n = w2 + x2 + y2 + z2
    x = x / n
    y = y / n
    z = z / n
    w = w / n
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.tensor([1 - 2*y2 - 2*z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, 1 - 2*x2 - 2*z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, 1 - 2*x2 - 2*y2]).reshape(  3, 3)
    return rotMat

def read_json_file(path):
    '''
    Read json file
    '''
    with open(path) as fp:
        json_data = json.load(fp)

    return json_data


def pose_2_matrix(pose):
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
    return  transformation_mat #@ flip_x


def pose_dict_to_numpy(pose):
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

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    o3d.visualization.draw_geometries([source_temp, target_temp])

def display_image(image):


    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def photometric_error(gt , pred):
    mask = generate_mask_NOCS(gt)

    difference = (gt - pred)

    difference[:,:,0] *= mask
    difference[:,:,1] *= mask
    difference[:,:,2] *= mask

    difference_squared = np.square(difference)
    difference_root = np.sqrt(difference_squared)
    return np.mean(difference_root)
    

def generate_mask_NOCS(nocs_map):
    '''
    Function to extract mask from NOCS map
    '''
    
    white = np.ones(nocs_map.shape)*1.0
    white = np.array([1, 1, 1])
    image_mask = np.abs(nocs_map[:,:,:3] - white).mean(axis=2) > 0.15
    

    return image_mask

def read_image(nocs_image_path):
    '''
    Reading NOCS image
    '''
    nocs_map = cv2.imread(nocs_image_path) 
    nocs_map = cv2.cvtColor(nocs_map, cv2.COLOR_BGR2RGB) / 255.0
    # print(nocs_map.shape)
    return nocs_map

def visualize_nocs_map(nocs_map, nm, mask=None, image = None):
    '''
    Plots 3D point cloud from nocs map
    Arguments:
        nocs_map - [H x W x 3] - NOCS map for image
    
    Returns: 
        None
    '''
    
    h, w = nocs_map.shape[:2]
    nocs_mask = mask#generate_mask_NOCS(nm)
    # print(np.unique(nocs_mask))
    # display_image(nocs_mask / 255.0)
    # plt.imshow(nocs_mask)
    # plt.show()
    nocs_mask_cloud = np.reshape(nocs_mask, (h*w))
    # print(nocs_mask_cloud.shape)
    nocs_cloud = np.reshape(nocs_map, (h*w, 3))
    # nocs_cloud = np.reshape(nocs_map, (3, h*w))

    nocs_cloud = nocs_cloud[nocs_mask_cloud == 1.0, :]
    colors = nocs_cloud

    if image is not None:
        image_cloud = np.reshape(image, (h*w, 3))
        image_cloud = image_cloud[nocs_mask_cloud == 1.0, :]
        colors = image_cloud 
           
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nocs_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0])

    return pcd
    
    


if __name__ == "__main__":

    # Ground Truth NOCS files

    # Cars
    gt_nocs            = sorted(glob.glob("../../../../14102020/cars/ValSet/val/*/*/*_NOXRayTL_00.png"))
    gt_depth            = sorted(glob.glob("../../../../14102020/cars/ValSet/val/*/*/*_Depth_00.exr"))
    gt_color            = sorted(glob.glob("../../../../14102020/cars/ValSet/val/*/*/*_Color_00.png"))
    gt_mask             = sorted(glob.glob("../../../../14102020/cars/ValSet/val/*/*/*_Mask_00.png"))
    gt_pose             = sorted(glob.glob("../../../../14102020/cars/ValSet/val/*/*/*_CameraPose.json"))

    pred_nocs            = sorted(glob.glob("/home/aadilmehdi/RRC/WSNOCS/14102020/cars/p1/pipeline1/*/*_nocs01.png"))
    pred_depth            = sorted(glob.glob("/home/aadilmehdi/RRC/WSNOCS/14102020/cars/p1/pipeline1/*/*_depth.tiff"))
    pred_color            = sorted(glob.glob("/home/aadilmehdi/RRC/WSNOCS/14102020/cars/p1/pipeline1/*/*_image.jpg"))
    pred_mask             = sorted(glob.glob("/home/aadilmehdi/RRC/WSNOCS/14102020/cars/p1/pipeline1/*/*_mask.jpg"))
    pred_ply             = sorted(glob.glob("/home/aadilmehdi/RRC/WSNOCS/14102020/cars/p1/pipeline1/*/*_point_cloud.ply"))

    coss = []

    # Camera Matrix
    K = camera_matrix(888.88, 1000.0, 320, 240)
    from tqdm import tqdm
    for i in tqdm(range(len(gt_nocs))):

        # Read the color image
        image_view = cv2.imread(gt_color[i])
        image_view = cv2.cvtColor(image_view, cv2.COLOR_BGR2RGB) / 255.0
        image = cv2.resize(image_view, (640, 480))

        # Read the depth image
        depth = imageio.imread(gt_depth[i])[:, :, 0]

        # Read the nocs image
        nocs = read_image(gt_nocs[i])

        # # Read the pose
        # pose = read_json_file(gt_pose[i])
        # print("Pose", pose)
        # pose = torch.from_numpy(pose_dict_to_numpy(pose))
        # print("Pose", pose)
        # # pose = pose_2_matrix(pose)
        # rot = quat2mat(pose[3],pose[4],pose[5],pose[6])
        # print("Rot", rot)
        
        # Get the mask
        mask = generate_mask_NOCS(nocs)

        # Get the depth point cloud
        depth_point_cloud, depth_pcd = save_point_cloud(K, image, mask, depth, num=0, output_directory='./',depth_tiff=None)
        depth_points = np.asarray(depth_pcd.points) 
        depth_points = depth_points - np.mean(depth_points, axis=0)
        depth_pcd.points = o3d.utility.Vector3dVector(depth_points)

        # Get the nocs point cloud
        nocs_pcd          = visualize_nocs_map(nocs, nocs, mask=mask)
        nocs_points       = np.asarray(nocs_pcd.points) - 0.5
        nocs_pcd.points = o3d.utility.Vector3dVector(nocs_points)


        # Estimate Umeyama Alignment 
        Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(nocs_points.T, depth_points.T)

        # ICP Alignment
        reg_p2p = o3d.registration.registration_icp(
                depth_pcd, nocs_pcd, 0.2, OutTransform,
                o3d.registration.TransformationEstimationPointToPoint())

        # draw_registration_result(depth_pcd, nocs_pcd, reg_p2p.transformation) 
        # print("GT: ", reg_p2p.transformation)        

        gt_transform = reg_p2p.transformation


        ###########################################################################################################33

        # Read the color image
        image_view = cv2.imread(pred_color[i])
        image_view = cv2.cvtColor(image_view, cv2.COLOR_BGR2RGB) / 255.0
        image = cv2.resize(image_view, (640, 480))

        # Read the depth image
        depth = imageio.imread(pred_depth[i])#[:, :, 0]

        # Read the nocs image
        nocs = read_image(pred_nocs[i])
        
        # Get the mask
        mask = generate_mask_NOCS(nocs)

        # Get the nocs point cloud
        nocs_pcd          = visualize_nocs_map(nocs, nocs, mask=mask)
        nocs_points       = np.asarray(nocs_pcd.points) - 0.5


        # Get the depth point cloud
        # depth_point_cloud, depth_pcd = save_point_cloud(K, image, mask, depth, num=0, output_directory='./',depth_tiff=1)
        depth_pcd_file = pred_ply[i]
        depth_pcd = o3d.io.read_point_cloud(depth_pcd_file)
        depth_points = np.asarray(depth_pcd.points) 
        depth_points = depth_points - np.mean(depth_points, axis=0)

        m_dim = len(nocs_points)
        if len(depth_points) < m_dim:
            m_dim = len(depth_points)

        nocs_points = nocs_points[:m_dim, :]
        depth_points = depth_points[:m_dim, :]

        nocs_pcd.points = o3d.utility.Vector3dVector(nocs_points)
        depth_pcd.points = o3d.utility.Vector3dVector(depth_points)


        # Estimate Umeyama Alignment 
        Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(nocs_points.T, depth_points.T)

        depth_pcd.points = o3d.utility.Vector3dVector(depth_points)

        # draw_registration_result(depth_pcd, nocs_pcd, OutTransform)

        # ICP Alignment
        reg_p2p = o3d.registration.registration_icp(
                depth_pcd, nocs_pcd, 0.2, OutTransform,
                o3d.registration.TransformationEstimationPointToPoint())


        pred_transform = reg_p2p.transformation

        # draw_registration_result(depth_pcd, nocs_pcd, reg_p2p.transformation)

        unit_vec = np.ones((3,1))
        unit_vec /= np.linalg.norm(unit_vec)
        print("Unit Vec", unit_vec)


        gt_rot = gt_transform[:3,:3]  
        pred_rot = pred_transform[:3,:3]  

        print(gt_rot)
        print(pred_rot)

        gt_vec = np.squeeze((gt_rot @ unit_vec).T)
        pred_vec = np.squeeze((pred_rot @ unit_vec).T)

        print(f"gt_vec {gt_vec}")
        print(f"pred_vec {pred_vec}")

        d_prod = np.dot(gt_vec, pred_vec) / (np.linalg.norm(gt_vec) * np.linalg.norm(pred_vec))
        coss.append( np.degrees(np.arccos(d_prod)))
        print("Cos", np.degrees(np.arccos(d_prod)))

        fig = plt.figure(constrained_layout=True, figsize=(16,8))
        gs = GridSpec(1, 1, figure=fig)
        ax2 = fig.add_subplot(gs[0, 0])
        ax2.hist(coss, 20, density=False, histtype='stepfilled', facecolor='orange', alpha=0.75)
        ax2.set_title("mAP")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Frequency")
        plt.savefig(f"mAP Cars.png")

        coss_temp = np.array(coss)
        print("Mean:",np.mean(coss_temp, axis=0))
        print("StD:",np.std(coss_temp, axis=0))

    coss = np.array(coss)
    print("Mean:",np.mean(coss, axis=0))
    print("StD:",np.std(coss, axis=0))

    fig = plt.figure(constrained_layout=True, figsize=(16,8))
    gs = GridSpec(1, 1, figure=fig)
    ax2 = fig.add_subplot(gs[0, 0])
    ax2.hist(coss, 20, density=False, histtype='stepfilled', facecolor='orange', alpha=0.75)
    ax2.set_title("mAP")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Frequency")
    plt.savefig(f"mAP Cars.png")
