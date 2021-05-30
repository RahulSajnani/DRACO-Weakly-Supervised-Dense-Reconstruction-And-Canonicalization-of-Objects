import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from PIL import Image
import json
from mpl_toolkits.mplot3d import Axes3D as mpl_3D
# import pptk
import glob
import os

FLOAT_EPS = np.finfo(np.float).eps
# def camera_matrix(focal, c_x, c_y):
#     '''
#     Constructs camera matrix
#     '''

#     K = np.array([[ focal,      0,     c_x],
#                   [     0,  focal,     c_y],
#                   [     0,      0,       1]])
    
#     return K


def camera_matrix(focal_x, focal_y, c_x, c_y):
    '''
    Constructs camera matrix
    '''

    K = np.array([[ focal_x,      0,     c_x],
                  [     0,  focal_y,     c_y],
                  [     0,      0,       1]])

    return K
# Points generator
def get_grid(x, y):
    '''
    Get index grid from image
    '''
    # coords = np.indices((x, y)).reshape(2, -1)
    # return np.vstack((coords, np.ones(coords.shape[1])))

    y_i, x_i = np.indices((x, y))
    coords = np.stack([x_i, y_i, np.ones_like(x_i)], axis = -1).reshape(x*y, 3)
    # coords = np.indices((x, y)).reshape(2, -1)
    # return np.vstack((coords, np.ones(coords.shape[1])))
    print(coords)
    return coords.T

def depth_decode(depth_image):

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

    depth_single_channel
    depth_vector = depth_single_channel.reshape(1, -1)
    
    return depth_single_channel, depth_vector 

def extract_depth_tiff(depth_tiff):

    '''
    Extract depth from tiff image
    '''

    depth = np.array(depth_tiff)

    depth = 10 * (1 - depth)
    depth = depth.reshape(1, -1)
    return depth

def depth_2_point_cloud(invK, image, depth_image, depth_tiff=None):
    '''
    Convert depth map to point cloud

    '''
    points_hom = get_grid(image.shape[0], image.shape[1])
    


    if depth_tiff != None:
        print('tiff\n')
        depth = extract_depth_tiff(depth_tiff)
    else:
        depth = extract_depth(depth_image)
    # depth_map, depth = depth_decode(depth_image)
    
    print(np.min(depth), np.max(depth[depth<30]))
    point_3D = invK @ points_hom
    point_3D = point_3D / point_3D[2, :]
    point_3D = point_3D * depth
    
    return point_3D
    
def save_point_cloud(K, image, mask, depth, num, output_directory, depth_tiff = None):
    '''
    Save point cloud given depth map
    '''

    directory = '%s/' % output_directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    image_colors = image.reshape(-1, 3)
    # print(image_colors.shape)
    invK = np.linalg.inv(K)
    # invK[0,0] *= 1
    print(invK)
    point_cloud = depth_2_point_cloud(invK, image, depth, depth_tiff)   
    # point_cloud[0, :] *= -1
    mask = mask.reshape(-1, 1)
    mask = mask > 0.5
    # print(mask.shape)
    image_colors = image_colors[mask[:, 0], :]
    point_cloud = point_cloud[:, mask[:, 0]]
    # image_colors = image_colors[point_cloud[2,:] < 30, :]
    # point_cloud = point_cloud[:, point_cloud[2,:] < 30]
    
    image_colors = image_colors[point_cloud[2,:] < 5, :]
    point_cloud = point_cloud[:, point_cloud[2,:] < 5]
    

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
    ply_name = "%s/frame_%06d_point_cloud.ply" % (output_directory, num)
    o3d.io.write_point_cloud(ply_name, pcd)
    return point_cloud, pcd

def extract_depth(depth_map):

    '''
    Get a depth vector from image
    '''

    depth = depth_map.reshape(1, -1)
    depth = 8 - 8 * depth
    # depth = 1 / (depth + 0.001)
    
    return depth

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
    
def pose_to_transformation(pose):

    '''
    Convert poses to transformation matrix
    '''


    temp_0 = pose[0]
    temp_1 = pose[1]
    temp_2 = pose[2]

    # quaternions
    temp_x = pose[3]
    temp_y = pose[4]
    temp_z = pose[5]
    temp_w = pose[6]

    pose[4:6] *= -1
    pose[0] *= -1

    # print(pose)
    rot_mat = quat2mat(pose[3:])
    translation_vector = np.array([[pose[0]], [pose[1]], [pose[2]]]) / 1000  
    print(translation_vector)
    # translation_offset = np.array([[2.25], [-1.25], [0.5]])
    # translation_offset = np.array([[0.0], [-0.5], [0.0]])
    
    rot_mat_2 = np.array([[ 0,  1, 0, 0],
                          [-1,  0, 0, 0],
                          [ 0,  0, 1, 0],
                          [ 0,  0, 0, 1]])
    
    flip_x = np.eye(4)
    flip_x[0, 0] *= -1

    trans = flip_x @ rot_mat_2
    translation_offset = np.ones((3,1)) * 1


    # different transformation matrix
    transformation_mat = np.vstack((np.hstack((rot_mat, translation_vector + 0.5 ) ), np.array([0, 0, 0, 1]))) 
    
    # transformation_mat = np.vstack((np.hstack((rot_mat.T,  rot_mat.T @ translation_vector)), np.array([0, 0, 0, 1])))
    
    # transformation_mat = np.vstack((np.hstack((rot_mat.T,   (translation_vector) )), np.array([0, 0, 0, 1])))
    
    # translation_offset = -np.array([[1.0], [1.0], [2.5]])
    
    # transformation_mat = np.vstack((np.hstack((rot_mat.T,  rot_mat.T @ (translation_offset)  + translation_vector)), np.array([0, 0, 0, 1])))
    
    print(transformation_mat.shape)
    return transformation_mat @ trans



def custom_draw_geometry(pcd):

    vis = o3d.visualization.Visualizer()
    
    vis.create_window()
    pcd_origin = o3d.geometry.PointCloud()
    pcd_origin.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0],
                                                             [0, 0, 1],
                                                             [1, 0, 0]]))
    pcd_origin.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    vis.add_geometry(pcd)
    vis.add_geometry(pcd_origin)
    
    ctr = vis.get_view_control()
    print(ctr)
    vis.run()

def read_json_file(path):
    '''
    Read json file
    '''
    json_data = []

    with open(path) as fp:
        for json_object in fp:
            json_data.append(json.loads(json_object))
    
    return json_data

if __name__ == "__main__":
    
    # directory_name = './val_2/'
    # directory_name = './outputs_4/val/'
    # directory_name = './outputs_5/val/'
    
    # Relative path to input directory
    input_directory_name = './'
    # directory_name = './outputs_3'
    # directory_name = './'

    ### 161649
    output_directory_name = './point_clouds/' + input_directory_name


    list_image = glob.glob(input_directory_name + "*image.jpg")
    list_image.sort()
    # print(list_image)
    
    for files in list_image:

        i = files.split('/')[-1].split('_')[1]
        i = int(i)

        image_number = i
        image_name = 'frame_%06d_image.jpg' % image_number
        image_depth = 'frame_%06d_depth.jpg' % image_number
        image_mask = 'frame_%06d_mask.jpg' % image_number
        image_depth_tiff = 'frame_%06d_depth.tiff' % image_number
        
        
        # K = camera_matrix(617.1, 315, 242)
        K = camera_matrix(888.88, 1000.0, 320, 240)

        # K = camera_matrix(571, 319.5, 239.5)

        mask = cv2.imread(input_directory_name + image_mask,  cv2.IMREAD_GRAYSCALE) / 255.0
        image = cv2.imread(input_directory_name + image_name) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        depth = cv2.imread(input_directory_name + image_depth,  cv2.IMREAD_GRAYSCALE) / 255.0
        depth_tiff = Image.open(input_directory_name + image_depth_tiff)

        # SAVE POINT CLOUD
        point_cloud, pcd = save_point_cloud(K,image, mask, depth, image_number, output_directory_name, depth_tiff)


   
