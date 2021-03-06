import glob
#from kornia import *
from PIL import Image
from skimage.transform import resize
import sys, os
sys.path.append(os.path.abspath(os.path.join('./models')))
sys.path.append(os.path.abspath(os.path.join('./Data_Loaders')))
sys.path.append(os.path.abspath(os.path.join('./Loss_Functions')))
from resnet import ResNetUNet, ResNetUNet50, NOCS_decoder, NOCS_decoder_2
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms, utils
import torch
import tqdm
import gc
import matplotlib.pyplot as plt
import numpy as np
import cv2
import open3d as o3d
import nocs_generator
import imageio
import helper_functions

global scale_factor

def extract_depth(depth_map):

    '''
    Get a depth vector from image
    '''

    depth = depth_map.reshape(1, -1)
    # depth = 8 - 8 * depth
    # depth = 1 / (depth + 0.001)

    return depth

def extract_depth_tiff(depth_tiff):

    '''
    Extract depth from tiff image
    '''

    depth = np.array(depth_tiff)
    scale_factor = 20
    depth = scale_factor * (1.0 - depth)
    # depth = 8 - 8*depth
    # depth = 1/(3*depth + 0.001)
    depth = depth.reshape(1, -1)
    return depth

def camera_matrix(focal_x, focal_y, c_x, c_y):
    '''
    Constructs camera matrix
    '''

    K = np.array([[ focal_x,      0,     c_x],
                  [     0,  focal_y,     c_y],
                  [     0,      0,       1]])

    return K


def generate_mask_NOCS(nocs_map):
    '''
    Function to extract mask from NOCS map
    '''

    white = np.ones(nocs_map.shape)*1.0
    white = np.array([1, 1, 1])
    image_mask = np.abs(nocs_map[:,:,:3] - white).mean(axis=2) > 0.0


    return image_mask


def preprocess_image(image, dim = 64):
    '''
    Function to preprocess image
    '''
    h, w, c = image.shape
    image_out = resize(image, (dim, dim))[:, :, :3]

    return image_out, {"scale_y": h / dim, "scale_x": w / dim}

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
    #print(coords)
    return coords.T

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

    directory = output_directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    image_colors = image.reshape(-1, 3)
    # print(image_colors.shape)
    invK = np.linalg.inv(K)
    # invK[0,0] *= 1
    #print(invK)
    point_cloud = depth_2_point_cloud(invK, image, depth, depth_tiff)
    # point_cloud[0, :] *= -1
    mask = mask.reshape(-1, 1)
    mask = mask > 0.5
    image_colors = image_colors[mask[:, 0], :]
    point_cloud = point_cloud[:, mask[:, 0]]
    # image_colors = image_colors[point_cloud[2,:] < 10, :]
    # point_cloud = point_cloud[:, point_cloud[2,:] < 10]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.T)
    pcd.colors = o3d.utility.Vector3dVector(image_colors)

    # ply_name = "%s/frame_%06d_point_cloud.ply" % (output_directory, num)
    # o3d.io.write_point_cloud(ply_name, pcd)
    return point_cloud, pcd


if __name__ == "__main__":

    ################################# Argument Parser #####################################

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", help = "Path to images", required=True)
    parser.add_argument("--model", help = "Model weights", required=False)
    parser.add_argument("--output", help = "Output directory path", required=False)
    parser.add_argument("--normalize", required=False, type=int)


    parser.add_argument("--depth_scale", help = "sigmoid 2 depth scale", type=int, default=20)

    args = parser.parse_args()

    #######################################################################################

    # directory_save = args.output
    # directory_save = os.path.join(directory_save, "")

    K = camera_matrix(888.88, 1000.0, 320, 240)

    # if not os.path.exists(directory_save):
    #     os.makedirs(directory_save)

    scale_factor = args.depth_scale

    images_list = glob.glob(os.path.join(os.path.abspath(args.path), "**/frame_**_Color_00.png"))
    #images_list = glob.glob(os.path.join(args.path, "**/**.jpg"))


    images_list.sort()

    #print(images_list)

    # net = ResNetUNet50()
    # nocs_decoder = NOCS_decoder()

    # if torch.cuda.is_available():
    #     net.load_state_dict(torch.load(args.model))
    #     net.cuda()
    # else:
    #     net.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))


    # net.eval()
    #print(images_list)
    i = 0
    for image_name in images_list:

        image_view = cv2.imread(image_name)
        image_view = cv2.cvtColor(image_view, cv2.COLOR_BGR2RGB) / 255.0
        image_view = cv2.resize(image_view, (640, 480))
        sub_directory = image_name.split('/')[-2]

        image_number = image_name.split('_')[-3]

        kp_name = image_name.split(".")[0] + ".npy"
        kp_no = kp_name.split("_")[-3]
        print(kp_no)
        dir_full = os.path.join("/", *(image_name.split("/")[:-1]))
        kp_name = "/".join(image_name.split("/")[:-1]) + "/frame_%08d_KeyPoints.npy" % int(kp_no)

        cam_coord = os.path.join(dir_full, "frame_%06d_cam_coords.npy" % int(kp_no))
        rotation = os.path.join(dir_full, "frame_%06d_rotation.npy" % int(kp_no))
        depth_file = os.path.join(dir_full, "frame_%08d_Depth_00.exr" % int(kp_no))
        depth = imageio.imread(depth_file)[:, :, 0].astype('float')
        depth = torch.from_numpy(depth).unsqueeze(0)
        print(depth.shape)


        nocs_file = os.path.join(dir_full, "frame_%08d_NOXRayTL_00.png" % int(kp_no))
        nocs_map = imageio.imread(nocs_file)[:, :, :3] / 255.0
        nocs_mask = generate_mask_NOCS(nocs_map)
        nocs_mask = (np.rint(nocs_mask).astype("uint8") * 255)
        mask_file = os.path.join(dir_full, "frame_%08d_Mask_00.png" % int(kp_no))
        cv2.imwrite(mask_file, nocs_mask)

        nocs_mask = torch.from_numpy(nocs_mask).unsqueeze(0)

        ### READ MASK FRM NOCS AND CHAGE TO TENSOR




        cam_coord = np.load(cam_coord)
        rotation = np.load(rotation)
        #print(kp_name)
        kp_matrix = np.load(kp_name)

        # kp_save = directory_save + "frame_%06d_KeyPoints.npy" % i

        print(image_name)

        if torch.cuda.is_available():
            image_tensor = torch.from_numpy(image_view.transpose(2, 0, 1)).unsqueeze(0).cuda()
        else:
            image_tensor = torch.from_numpy(image_view.transpose(2, 0, 1)).unsqueeze(0)


        data_sample = {}

        data_sample["masks"] = nocs_mask ## Ground truth mask
        print(cam_coord.shape, rotation.shape, kp_matrix.shape)

        data_sample["keypoints"] = torch.from_numpy(kp_matrix).float()

        data_sample["c3dpo_cam_coords"] = torch.from_numpy(cam_coord)[0].float()

        data_sample["c3dpo_rotation"] = torch.from_numpy(rotation)[0].float()

        data_sample["intrinsics"] = torch.from_numpy(K).unsqueeze(0).float()

        nocs = nocs_generator.generate_nocs_single(data_sample, depth.unsqueeze(0).float())

        # depth = (output[0][0,0]).cpu().detach().numpy()
        # mask = output[1][0,0].cpu().detach().numpy()


        # mask = (mask > 0.5) * 1.0
        #break


        # print(nocs.shape, mask.shape)
        nocs = nocs.detach().cpu().numpy()

        nocs[:, nocs_mask[0] < 0.5] = 1.0
        nocs = nocs.transpose((1, 2, 0))
        nocs = (nocs * 255).astype("uint8")

        c3dpo_file = os.path.join(dir_full, "frame_%08d_C3DPO_00.png" % int(kp_no))
        print(c3dpo_file)
        plt.imsave(c3dpo_file, nocs)

        print(K)

