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

global scale_factor

def extract_depth_tiff(depth_tiff):

    '''
    Extract depth from tiff image
    '''

    depth = np.array(depth_tiff)

    depth = scale_factor * (1.0 - depth)
    #depth = 8 - 8*depth
    #depth = 1/(3*depth + 0.001)
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
    image_colors = image_colors[point_cloud[2,:] < 10, :]
    point_cloud = point_cloud[:, point_cloud[2,:] < 10]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.T)
    pcd.colors = o3d.utility.Vector3dVector(image_colors)

    ply_name = "%s/frame_%06d_point_cloud.ply" % (output_directory, num)
    o3d.io.write_point_cloud(ply_name, pcd)
    return point_cloud, pcd


if __name__ == "__main__":

    ################################# Argument Parser #####################################

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", help = "Path to images", required=True)
    parser.add_argument("--model", help = "Model weights", required=True)
    parser.add_argument("--output", help = "Output directory path", required=True)
    parser.add_argument("--normalize", required=True, type=int)
    parser.add_argument("--decoder", help = "Model decoder weights", required=True)


    parser.add_argument("--depth_scale", help = "sigmoid 2 depth scale", type=int, default=10)
    parser.add_argument("--multi_seq", help = "Multiple sequences", type=int, default=1)
    parser.add_argument("--kp", help = "Multiple sequences", type=int, default=0)

    args = parser.parse_args()

    #######################################################################################

    directory_save = args.output
    directory_save = os.path.join(directory_save, "")

    K = camera_matrix(888.88, 1000.0, 320, 240)

    if not os.path.exists(directory_save):
        os.makedirs(directory_save)

    scale_factor = args.depth_scale

    if args.multi_seq:
        images_list = glob.glob(os.path.join(args.path, "**/frame_**_Color_00.png"))
        #images_list = glob.glob(os.path.join(args.path, "**/**.jpg"))
    else:
        images_list = glob.glob(os.path.join(args.path, "frame_**_Color_00.png"))
        #images_list = glob.glob(os.path.join(args.path, "**.jpg"))

    images_list.sort()

    #print(images_list)

    net = ResNetUNet50()
    nocs_decoder = NOCS_decoder()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(args.model))
        net.cuda()
        nocs_decoder.load_state_dict(torch.load(args.decoder))
        nocs_decoder.cuda()
    else:
        net.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
        nocs_decoder.load_state_dict(torch.load(args.decoder, map_location=torch.device("cpu")))


    net.eval()
    #print(images_list)
    i = 0
    for image_name in images_list:


        image_view = cv2.imread(image_name)
        image_view = cv2.cvtColor(image_view, cv2.COLOR_BGR2RGB) / 255.0
        image_view = cv2.resize(image_view, (640, 480))

        if args.kp:
            kp_name = image_name.split(".")[0] + ".npy"
            kp_no = kp_name.split("_")[-3]
            #print(kp_name)
            kp_name = "/".join(image_name.split("/")[:-1]) + "/frame_%08d_KeyPoints.npy" % int(kp_no)
            #print(kp_name)
            kp_matrix = np.load(kp_name)
            kp_save = directory_save + "frame_%06d_KeyPoints.npy" % i
            print(kp_save)
            np.save(kp_save, kp_matrix)

        #image_view = cv2.resize(image_view, (440, 280))
        #image_view = cv2.copyMakeBorder(image_view, 100, 100, 100, 100, cv2.BORDER_CONSTANT)
        #image_view = plt.imread(image_name)[:, :, :3]
        print(image_name)
        #image_view, scale_dict = preprocess_image(image_view)

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

        output = net(image_tensor.float(), encoder = True)
        depth = (output[0][0,0]).cpu().detach().numpy()
        mask = output[1][0,0].cpu().detach().numpy()

        mask = (mask > 0.5) * 1.0

        nocs = (nocs_decoder(output[2]))
        print(nocs.shape, mask.shape)
        nocs = nocs.detach().cpu().numpy()

        nocs[0, :, mask < 0.5] = 1.0
        nocs = nocs[0].transpose((1, 2, 0))
        nocs = (nocs * 255).astype("uint8")
        depth = (depth * mask)
        im_depth = depth.astype('float32')
        im_depth_tiff = Image.fromarray(im_depth, 'F')

        save_point_cloud(K, image_view, mask, depth, i, directory_save, im_depth_tiff)

        image = (image_view*255).astype("uint8")
        #image = image_view
        #print(image.shape)
        mask = (mask*255).astype("uint8")
        depth = (depth*255).astype(int)
        #image = (image*255).astype(int)



        #################################### SAVING FILES ###########################################
        print("Saving")
        mask_name = directory_save + 'frame_%06d_mask.jpg' % i
        image_name = directory_save +'frame_%06d_image.jpg' % i
        depth_name = directory_save +'frame_%06d_depth.jpg' % i
        nocs_name = directory_save + "frame_%06d_nocs.jpg" % i
        depth_tiff_name = directory_save +'frame_%06d_depth.tiff' % i

        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        plt.imsave(image_name, image)
        plt.imsave(nocs_name, nocs)
        #cv2.imwrite(image_name, image)
        cv2.imwrite(depth_name, depth)
        cv2.imwrite(mask_name, mask)
        im_depth_tiff.save(depth_tiff_name)
        #############################################################################################
        i = i + 1
