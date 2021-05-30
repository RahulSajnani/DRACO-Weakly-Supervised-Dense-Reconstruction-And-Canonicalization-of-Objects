import open3d as o3d
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def display_image(image):


    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def transform_nocs(nocs_map, rotation_matrix):

    '''
    Rotates NOCS point cloud

    Arguments:
        nocs_map - [H x W x 3] - NOCS map for image
        rotation_matrix - [3 x 3] - rotation matrix to rotate NOCS map
    Returns:
        nocs_cloud_image - [H x W x 3] - rotates NOCS frame
    '''

    h, w = nocs_map.shape[:2]
    nocs_mask = generate_mask_NOCS(nocs_map)
    nocs_cloud = np.reshape(nocs_map, (h*w, 3)) # [N, 3]
    nocs_cloud = rotation_matrix @ np.concatenate((nocs_cloud.T - 0.5, np.ones((1,h*w))), axis=0)
    nocs_cloud[0,:]/=nocs_cloud[3,:]
    nocs_cloud[1,:]/=nocs_cloud[3,:]
    nocs_cloud[2,:]/=nocs_cloud[3,:]
    nocs_cloud[3,:]/=nocs_cloud[3,:]
    nocs_cloud = nocs_cloud.T
    nocs_cloud[:,0:3] += 0.5
    print(nocs_cloud)
    nocs_cloud_image = np.reshape(nocs_cloud[:,:3], (h, w, 3))
    nocs_cloud_image[nocs_mask < 0.5, :] = 1.0
    nocs_cloud_image = nocs_cloud_image.clip(min=0, max=1)

    return nocs_cloud_image


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

    return nocs_map

def rotate_nocs(nocs_map, rotation_matrix):

    '''
    Rotates NOCS point cloud

    Arguments:
        nocs_map - [H x W x 3] - NOCS map for image
        rotation_matrix - [3 x 3] - rotation matrix to rotate NOCS map
    Returns: 
        nocs_cloud_image - [H x W x 3] - rotates NOCS frame
    '''
    
    h, w = nocs_map.shape[:2]
    nocs_mask = generate_mask_NOCS(nocs_map)
    nocs_cloud = np.reshape(nocs_map, (h*w, 3)) # [N, 3]
    nocs_cloud = (rotation_matrix @ (nocs_cloud.T - 0.5)).T + 0.5
    nocs_cloud_image = np.reshape(nocs_cloud, (h, w, 3))
    nocs_cloud_image[nocs_mask < 0.5, :] = 1.0
    nocs_cloud_image = nocs_cloud_image.clip(min=0, max=1)    
    
    return nocs_cloud_image

def visualize_nocs_map(nocs_map, image = None):
    '''
    Plots 3D point cloud from nocs map

    Arguments:
        nocs_map - [H x W x 3] - NOCS map for image
    
    Returns: 
        None
    '''
    
    h, w = nocs_map.shape[:2]
    nocs_mask = generate_mask_NOCS(nocs_map)
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
    vis_obj = o3d.visualization
    # print(vis_obj.RenderOption.show_coordinate_frame)
    # vis_obj.RenderOption.show_coordinate_frame.setter("True")
    # print(vis_obj["light_on"])
    # vis_obj.RenderOption.show_coordinate_frame = True
    
  
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

    vis_obj.draw_geometries([pcd, line_set])
    # o3d.visualization.draw_geometries([pcd])


    return pcd
    
if __name__ == "__main__":
    # print("hi")
    nocs_image_path = sys.argv[1]
    image = None
    
    if len(sys.argv) > 2:
        image_path = sys.argv[2]
        image = read_image(image_path)


    nocs_map = read_image(nocs_image_path)
    pcd = visualize_nocs_map(nocs_map, image)
    # a = o3d.geometry.OrientedBoundingBox()  
    # a.create_from_points(pcd.points)
    # print(np.asarray(a.get_box_points()))