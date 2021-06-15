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
from trainers import DRACO_phase_2 as training_model
from omegaconf import OmegaConf


def camera_matrix(focal_x, focal_y, c_x, c_y):
    '''
    Constructs camera matrix
    '''

    K = np.array([[ focal_x,      0,     c_x],
                  [     0,  focal_y,     c_y],
                  [     0,      0,       1]])

    return K

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

    y_i, x_i = np.indices((x, y))
    coords = np.stack([x_i, y_i, np.ones_like(x_i)], axis = -1).reshape(x*y, 3)

    return coords.T

def depth_2_point_cloud(invK, image, depth_image, depth_tiff=None):
    '''
    Convert depth map to point cloud

    '''
    points_hom = get_grid(image.shape[0], image.shape[1])

    depth = depth_image.reshape(1, -1)
    point_3D = invK @ points_hom
    point_3D = point_3D / point_3D[2, :]
    point_3D = point_3D * depth

    return point_3D

def save_point_cloud(K, image, mask, depth, ply_name, depth_tiff = None):
    '''
    Save point cloud given depth map
    '''



    image_colors = image.reshape(-1, 3)
    
    invK = np.linalg.inv(K)
    point_cloud = depth_2_point_cloud(invK, image, depth, depth_tiff)

    mask = mask.reshape(-1, 1)
    mask = mask > 0.5
    image_colors = image_colors[mask[:, 0], :]
    point_cloud = point_cloud[:, mask[:, 0]]
    image_colors = image_colors[point_cloud[2,:] < 10, :]
    point_cloud = point_cloud[:, point_cloud[2,:] < 10]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.T)
    pcd.colors = o3d.utility.Vector3dVector(image_colors)

    o3d.io.write_point_cloud(ply_name, pcd)
    return point_cloud, pcd



if __name__ == "__main__":

    ################################# Argument Parser #####################################

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path", help = "Path to images", required=True)
    parser.add_argument("--model", help = "Model weights", required=True)
    parser.add_argument("--output", help = "Output directory path", required=True)



    parser.add_argument("--normalize", required=False, type=int, default=1)
    parser.add_argument("--real", help = "Real images?", required=False, default = 0)
    # parser.add_argument("--multi_seq", help = "Multiple sequences", type=int, default=1)

    args = parser.parse_args()

    #######################################################################################
    #cfgs = OmegaConf.load('cfgs/config_DRACO.yaml')
    print(args.model)

    directory_save = args.output
    directory_save = os.path.join(directory_save, "")

    K = camera_matrix(888.88, 1000.0, 320, 240)

    if not os.path.exists(directory_save):
        os.makedirs(directory_save)


    images_list = glob.glob(os.path.join(args.path, "**/frame_**_Color_00.png"))
    
    if args.real:
        images_list = glob.glob(os.path.join(args.path, "**/**.jpg"))

    if len(images_list) == 0:
        images_list = glob.glob(os.path.join(args.path, "frame_**_Color_00.png"))
        if args.real:
            images_list = glob.glob(os.path.join(args.path, "**.jpg"))

    images_list.sort()


    net = training_model.load_from_checkpoint(args.model)
    net.eval()

    if torch.cuda.is_available():
        net.cuda()



    for image_name in images_list:
        
        directory, tail = os.path.split(image_name)
        name_without_ext, ext = os.path.splitext(tail)
        print(name_without_ext)

        sub_directory_save = os.path.join(directory_save + image_name.split('/')[-2], "")
        if not os.path.exists(sub_directory_save):
            os.makedirs(sub_directory_save)
        
        image_view = cv2.imread(image_name)
        image_view = cv2.cvtColor(image_view, cv2.COLOR_BGR2RGB) / 255.0
        image_view = cv2.resize(image_view, (640, 480))

        
        if args.real:
            image_view = cv2.resize(image_view, (440, 280))
            image_view = cv2.copyMakeBorder(image_view, 100, 100, 100, 100, cv2.BORDER_CONSTANT)
        
        if torch.cuda.is_available():
            image_tensor = torch.from_numpy(image_view.transpose(2, 0, 1)).unsqueeze(0).cuda()
        else:
            image_tensor = torch.from_numpy(image_view.transpose(2, 0, 1)).unsqueeze(0)

        if args.normalize:
            print("NORMALIZE")
            normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225],
                                                            inplace=True)
            image_tensor = normalize_transform(image_tensor[0]).unsqueeze(0)

        output = net.test_pass(image_tensor.float())
        depth = (output[0][0,0]).cpu().detach().numpy()
        mask = output[1][0,0].cpu().detach().numpy()

        mask = (mask > 0.5) * 1.0

        nocs = (net.nocs_decoder(output[2]))
        nocs = nocs.detach().cpu().numpy()

        nocs[0, :, mask < 0.5] = 1.0
        nocs = nocs[0].transpose((1, 2, 0))
        nocs = (nocs * 255).astype("uint8")
        depth = (depth * mask)


        #################################### SAVING FILES ###########################################

        print("Saving")
        im_depth = depth.astype('float32')
        im_depth_tiff = Image.fromarray(im_depth, 'F')

        ply_name = sub_directory_save + name_without_ext + "_pointcloud.ply"
        save_point_cloud(K, image_view, mask, depth, ply_name, im_depth_tiff)
        image = (image_view*255).astype("uint8")
        mask = (mask*255).astype("uint8")


        mask_name = sub_directory_save + name_without_ext + "_mask.jpg" 
        nocs_name = sub_directory_save + name_without_ext + "_nocs.jpg"
        depth_tiff_name = sub_directory_save + name_without_ext + "_depth.tiff"

        plt.imsave(image_name, image)
        plt.imsave(nocs_name, nocs)
        cv2.imwrite(mask_name, mask)
        im_depth_tiff.save(depth_tiff_name)
        #############################################################################################
