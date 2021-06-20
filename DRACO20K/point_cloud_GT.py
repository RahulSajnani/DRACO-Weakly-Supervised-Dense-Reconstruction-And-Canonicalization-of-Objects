from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os, sys
import json
FLOAT_EPS = np.finfo(np.float).eps
#sys.path.append(os.path.abspath(os.path.join('../')))
# from gen_point_cloud import *
import imageio
import torch

import mathutils


def quat2mat(quat):
    
    x, y, z, w = quat
    Nq = w*w + x*x + y*y + z*z
    if Nq < FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

def get_grid(x, y):
    '''
    Get index grid from image
    '''
    y_i, x_i = np.indices((x, y))
    coords = np.stack([x_i, y_i, np.ones_like(x_i)], axis = -1).reshape(x*y, 3)
    # coords = np.indices((x, y)).reshape(2, -1)
    # return np.vstack((coords, np.ones(coords.shape[1])))
    print(coords)
    return coords.T

def camera_matrix_2(focal, c_x, c_y):
    '''
    Constructs camera matrix
    '''

    K = np.array([[ focal,      0,     -c_x],
                  [     0,  focal,     -c_y],
                  [     0,      0,       1]])
    
    return K

def depth_decode_2(depth_image):

    # depth_image = np.array(depth_image)
    # # first 16 bits (first 2 channels) are 16-bit depth
    # R is the 8 LSB and G are the others
    depth_image_16 = depth_image[:,:,[1, 0]]
    # B are 8-bit version
    depth_image_8 = depth_image[:,:,2]
    # plt.imshow(depth_image_8)
    # plt.show()

    # last 8 are empty
    depth_single_channel = np.zeros((depth_image_16.shape[0], depth_image_16.shape[1]))
    # convert 16 bit to actual depth values
    for i in range(depth_single_channel.shape[0]):
        for j in range(depth_single_channel.shape[1]):
            bit_str = '{0:08b}'.format(depth_image_16[i, j, 0]) + '{0:08b}'.format(depth_image_16[i, j, 1])
            depth_single_channel[i, j] = int(bit_str, 2)

    depth_single_channel /= 1000
    print(np.min(depth_single_channel))

    # depth_single_channel
    depth_vector = depth_single_channel#.reshape(1, -1)
    depth_vector = depth_vector.reshape(1,-1)
    return depth_single_channel, depth_vector 


def read_json_file(path):
    '''
    Read json file
    '''
    json_data = []

    with open(path) as fp:
        for json_object in fp:
            json_data.append(json.loads(json_object))
    
    return json_data

def camera_matrix(focal_x, focal_y, c_x, c_y):
    '''
    Constructs camera matrix
    '''

    K = np.array([[ focal_x,      0,     c_x],
                  [     0,  focal_y,     c_y],
                  [     0,      0,       1]])
    
    return K

def json_to_numpy(pose):
    pose = np.array([pose['position']['x'], 
                     pose['position']['y'], 
                     pose['position']['z'],
                     pose['rotation']['x'], 
                     pose['rotation']['y'], 
                     pose['rotation']['z'], 
                     pose['rotation']['w'] 
                     ])
    return pose

def pose_2_transformation(pose):

    '''
    Convert poses to transformation matrix
    '''

  
    # pose[5] *= -1
    rot_mat_2 = np.array([[ 1,  0, 0, 0],
                          [ 0,  0, 1, 0],
                          [ 0, -1, 0, 0],
                          [ 0,  0, 0, 1]])
    
    
    flip_y = np.eye(4)
    flip_y[1,1] *= -1
    flip_y[2,2] *= -1

    flip_x = np.eye(4)
    flip_x[0, 0] *= -1
    flip_x[1, 1] *= -1

    rot_mat = quat2mat(pose[3:])
    print(rot_mat)
    # rot_mat = np.array(mathutils.Quaternion((pose[6], pose[3], pose[4], pose[5])).to_matrix())

    print(mathutils.Quaternion((pose[6], pose[3], pose[4], pose[5])).to_matrix())
    translation_vector = np.array([[pose[0]], [pose[1]], [pose[2]]]) # / 1000


    transformation_mat = np.vstack((np.hstack((rot_mat,   translation_vector  ) ), np.array([0, 0, 0, 1])))

    return transformation_mat @ flip_y 

    # return np.linalg.inv(transformation_mat) @ flip_x @ flip_y 



def save_point_cloud_2(K, image, mask, depth):
    '''
    Save point cloud given depth map
    '''

    image_colors = image.reshape(-1, 3)
    # print(image_colors.shape)
    invK = np.linalg.inv(K)
    # invK[0,0] *= 1
    print(invK)
    point_cloud = depth_2_point_cloud_2(invK, image, depth)   
    # point_cloud[0, :] *= -1
    mask = mask.reshape(-1, 1)
    mask = mask > 0.5
    # print(mask.shape)
    image_colors = image_colors[mask[:, 0], :]
    point_cloud = point_cloud[:, mask[:, 0]]
    # image_colors = image_colors[point_cloud[2,:] < 30, :]
    # point_cloud = point_cloud[:, point_cloud[2,:] < 30]
    
    image_colors = image_colors[point_cloud[2,:] < 10, :]
    point_cloud = point_cloud[:, point_cloud[2,:] < 10]
    

    # image_colors = image_colors[point_cloud[2,:] < 300, :]
    # point_cloud = point_cloud[:, point_cloud[2,:] < 300]
    
    # point_cloud[, :] = -point_cloud[2,:]
    # point_cloud[2, :] *= 10
    # print(np.min(point_cloud[2, :]), np.max(point_cloud[2,:]))
    # print(point_cloud.shape)
    
    
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.T)
    pcd.colors = o3d.utility.Vector3dVector(image_colors)
    # print(pcd.colors)
    o3d.io.write_point_cloud("./car.ply", pcd)
    return point_cloud, pcd


def depth_2_point_cloud_2(invK, image, depth_image):
    '''
    Convert depth map to point cloud

    '''
    points_hom = get_grid(image.shape[0], image.shape[1])
    depth = depth_image.reshape(1,-1)
    # depth_map, depth = depth_decode_2(depth_image)

    
    print(np.min(depth), np.max(depth[depth<30]))
    point_3D = invK @ points_hom
    point_3D = point_3D / point_3D[2, :]
    point_3D = point_3D * depth
    
    return point_3D

def get_exr_image(openexr_file):

    '''
    Converts openexr to python
    '''

    image = imageio.imread(openexr_file)

    image = image[:, :, 0]

    image[image == np.inf] = 30
    return image

def extract_pose(pose_path):

    pose_json = read_json_file(pose_path)
    pose = torch.from_numpy(json_to_numpy(pose_json[0])).unsqueeze(0)

    return pose


def get_image_pil(image_path):

    image = Image.open(image_path)
    image = np.array(image)
    
    return np.array(image)

if __name__ == "__main__":
    # directory = '/home/rahul/Robotics Research Centre/datasets/Multiview-NOCS/cars/'
    # seq = '27f7336ff6ace6d460c94cf1a48e53cb/
   
    directory = './seq_1/'

    seq = ''
    # directory = '../'
    # seq = ''

    image_number = 39

    increment = 1

    image_name = directory + seq + 'frame_%08d_Color_00.png' % image_number
    image_mask_1 = directory + seq + 'frame_%08d_Mask_00.png' % (image_number)
    
    # nocs_image_name = directory + seq + 'frame_%08d_NOXRayTL_00.png' % image_number
    pose_name = directory + seq + 'frame_%08d_CameraPose.json' % image_number
    depth_image_1 = directory + seq + 'frame_%08d_Depth_00.exr' % (image_number)
    
    pose_name_2 = directory + seq + 'frame_%08d_CameraPose.json' % (image_number + increment)
    # nocs_image_name_2 = directory + seq + 'frame_%08d_NOXRayTL_00.png' % (image_number + increment)
    image_mask_2 = directory + seq + 'frame_%08d_Mask_00.png' % (image_number + increment)
    image_name_2 = directory + seq + 'frame_%08d_Color_00.png' % (image_number + increment)
    depth_image_2 = directory + seq + 'frame_%08d_Depth_00.exr' % (image_number + increment)
    

    # Obtain depths
    depth_1 = get_exr_image(depth_image_1)
    depth_2 = get_exr_image(depth_image_2)

    # Obtain masks
    mask_1 = np.rint(np.array(imageio.imread(image_mask_1) / 65535.0))
    mask_2 = np.rint(np.array(imageio.imread(image_mask_2) / 65535.0))
    
    # depth_1[mask_1!= 1] = 0
    # plt.imshow(depth_1 / depth_1[mask_1 == 1].max(), 'gray')
    # plt.show()
    # print(depth_1[mask_1 == 1].min())
    

    # extracting image
    image = get_image_pil(image_name)
    image = image[:, :, :3] / 255.0
    

    image_2 = get_image_pil(image_name_2)
    image_2 = image_2[:, :, :3] / 255.0

    # extracting pose
    # pose_json = read_json_file(pose_name)
    # pose_source = torch.from_numpy(json_to_numpy(pose_json[0])).unsqueeze(0)
    
    pose_source = extract_pose(pose_name).squeeze()
    pose_dest = extract_pose(pose_name_2).squeeze()
    
    print(pose_source)
    # print(image.shape)
    # K = camera_matrix(617.1, 320, 240)
    K = camera_matrix(888.88, 1000, 320, 240)
    # K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    # K = camera_matrix(577.5, 319.5, 239.5)

    point_cloud, pcd_ego = save_point_cloud_2(K, image, mask_1, depth_1)

    point_cloud_homogenous = np.vstack((point_cloud, np.ones((1, point_cloud.shape[1]))))

    rot_mat_2 = np.array([[ 1,  0, 0, 0],
                          [ 0,  0, 1, 0],
                          [ 0,  -1, 0, 0],
                          [ 0,  0, 0, 1]])
    
    # rot_mat_1 = np.array([[ 1,  0, 0, -0.5],
    #                       [ 0,  1, 0, -0.5],
    #                       [ 0,  0, 1, 0.0],
    #                       [ 0,  0, 0, 1]])
    
    flip_y = np.eye(4)
    flip_y[1,1] *= -1
    flip_y[2,2] *= -1
    flip_x = np.eye(4)
    flip_x[0, 0] *= -1

    # # trans = flip_x @ rot_mat_2 

    flip_x[1, 1] *= -1
    point_cloud_dest =   flip_y @ point_cloud_homogenous

    # point_cloud_dest = trans @ point_cloud_homogenous

    pcd_ego.points = o3d.utility.Vector3dVector(point_cloud_dest[:3,:].T)
    o3d.io.write_point_cloud("./car_trans_correct.ply", pcd_ego)

    print('inv project')

    # print(np.min(point_cloud_dest[0, :]))
    # print(np.min(point_cloud_dest[1, :]))

    # pose_json = read_json_file(pose_name)
    # pose = json_to_numpy(pose_json[0])
    transformation_source = pose_2_transformation(pose_source)
    
    # pose_json_2 = read_json_file(pose_name_2)
    # pose = json_to_numpy(pose_json_2[0])
    transformation_dest = pose_2_transformation(pose_dest)
    



    # camera_frame_coords = np.linalg.inv(trans) @ np.linalg.inv(transformation_source) @ nocs_image_homogeneous

    # camera_frame_coords_2 = np.linalg.inv(transformation_dest) @ transformation_source @ camera_frame_coords
    camera_frame_coords_2 = np.linalg.inv(transformation_dest) @ transformation_source @ point_cloud_homogenous
    
    # camera_frame_coords = camera_frame_coords[:3, :].T
    
    
    
    # camera_frame_coords = camera_frame_coords_2[:3, :].T


    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(camera_frame_coords)
    pcd.colors = o3d.utility.Vector3dVector(image.reshape((-1, 3)))
    
    # o3d.io.write_point_cloud("./car_frame_1.ply", pcd)

    pcd_ego.points = o3d.utility.Vector3dVector(camera_frame_coords_2[:3,:].T)
    o3d.io.write_point_cloud("./car_frame_2.ply", pcd_ego)



    