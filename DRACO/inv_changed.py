# Code adapted from https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

from __future__ import division
from pytorch3d.ops.knn import knn_points
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import helper_functions
FLOAT_EPS = np.finfo(np.float).eps
pixel_coords = None
import kornia

from scipy.spatial.transform import Rotation as R

def preprocess_depth_output_2_point_cloud_all(depth_maps, masks, intrinsics):
    '''
    Pre process data for pose network

    Function mean subtracts the point cloud to bring it to origin and downsamples it to 2048 points
    '''

    batch_size, num_views, height, width  = depth_maps.size()

    depth_maps = helper_functions.sigmoid_2_depth(depth_maps)

    point_cloud_list_all_views = []
    rotated_point_cloud_list_all_views = []

    for view in range(num_views):

        src_camera_coords = pixel2cam(depth_maps[:, view].unsqueeze(0), intrinsics.inverse())
        src_camera_coords = src_camera_coords.reshape(batch_size, 3, height*width) # [B 3 H*W]

        if torch.cuda.is_available():
            random_rotation = torch.from_numpy(R.random(batch_size, random_state=1024).as_matrix()).cuda().float() # [B 3 3]
        else:
            random_rotation = torch.from_numpy(R.random(batch_size, random_state=1024).as_matrix()).float() # [B 3 3]

        point_cloud_list = []
        rotated_point_cloud_list = []
        masks_batch = masks[:, view]

        for i in range(batch_size):

            src_camera_coords_view = src_camera_coords[i] # [3 H*W]
            mask = masks_batch[i] # [H W]
            mask = mask.reshape(1, -1).squeeze() # [H*W]

            # Extracting the points only within mask region
            src_camera_coords_view = src_camera_coords_view[:, (mask == 1.0)]

            # Mean center value
            src_camera_coords_view = src_camera_coords_view - src_camera_coords_view.mean(axis = 1).unsqueeze(1).repeat(1, src_camera_coords_view.size(1)) #[3 masksize]

            # Downsample to 2048 points
            src_camera_coords_view = torch.nn.functional.interpolate(src_camera_coords_view.unsqueeze(0), size = 2048).squeeze(0)

            point_cloud_list.append(src_camera_coords_view)


        src_camera_coords_downsampled = torch.stack(point_cloud_list) # [B 3 2048]
        rot_src_camera_coords = random_rotation @ src_camera_coords_downsampled # [B 3 2048]

        point_cloud_list_all_views.append(src_camera_coords_downsampled)
        rotated_point_cloud_list_all_views.append(rot_src_camera_coords)

    camera_point_clouds_downsampled = torch.stack(point_cloud_list_all_views, dim = 1) # [B views 2048]
    rotated_camera_point_clouds_downsampled = torch.stack(rotated_point_cloud_list_all_views, dim = 1) # [B views 2048]

    return camera_point_clouds_downsampled, rotated_camera_point_clouds_downsampled


def preprocess_depth_output_2_point_cloud(depth_maps, masks_batch, intrinsics):
    '''
    Pre process data for pose network

    Function mean subtracts the point cloud to bring it to origin and downsamples it to 2048 points
    '''

    batch_size, _, height, width  = depth_maps.size()

    depth_maps = helper_functions.sigmoid_2_depth(depth_maps)
    src_camera_coords = pixel2cam(depth_maps[:, 0].unsqueeze(0), intrinsics.inverse())
    src_camera_coords = src_camera_coords.reshape(batch_size, 3, height*width) # [B 3 H*W]

    if torch.cuda.is_available():
        random_rotation = torch.from_numpy(R.random(batch_size, random_state=1024).as_matrix()).cuda().float() # [B 3 3]
    else:
        random_rotation = torch.from_numpy(R.random(batch_size, random_state=1024).as_matrix()).float() # [B 3 3]

    point_cloud_list = []
    rotated_point_cloud_list = []

    for i in range(batch_size):

        src_camera_coords_view = src_camera_coords[i] # [3 H*W]
        mask = masks_batch[i] # [H W]
        mask = mask.reshape(1, -1).squeeze() # [H*W]

        # Extracting the points only within mask region
        src_camera_coords_view = src_camera_coords_view[:, (mask == 1.0)]

        # mean center value
        src_camera_coords_view = src_camera_coords_view - src_camera_coords_view.mean(axis = 1).unsqueeze(1).repeat(1, src_camera_coords_view.size(1)) #[3 masksize]

        # Downsample to 2048 points
        src_camera_coords_view = torch.nn.functional.interpolate(src_camera_coords_view.unsqueeze(0), size = 2048).squeeze(0)

        point_cloud_list.append(src_camera_coords_view)

    src_camera_coords_downsampled = torch.stack(point_cloud_list) # [B 3 2048]
    rot_src_camera_coords = random_rotation @ src_camera_coords_downsampled # [B 3 2048]

    return src_camera_coords_downsampled, rot_src_camera_coords


def depth_decode(depth_image):

    # # first 16 bits (first 2 channels) are 16-bit depth
    # R is the 8 LSB and G are the others
    depth_image_16 = depth_image[:,:,[1, 0]]
    # B are 8-bit version
    depth_image_8 = depth_image[:,:,2]

    # last 8 are empty
    depth_single_channel = np.zeros((depth_image_16.shape[0], depth_image_16.shape[1]))
    # convert 16 bit to actual depth values
    for i in range(depth_single_channel.shape[0]):
        for j in range(depth_single_channel.shape[1]):
            bit_str = '{0:08b}'.format(depth_image_16[i, j, 0]) + '{0:08b}'.format(depth_image_16[i, j, 1])
            depth_single_channel[i, j] = int(bit_str, 2)

    return depth_single_channel

def set_id_grid(depth):
    global pixel_coords
    b, _, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth)
    #print("i_range",i_range.device)
    #print("j_range",j_range.device)
    #print("ones",ones.device)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1).type_as(depth)  # [1, 3, H, W]
    pixel_coords.to(depth.device)

def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr):
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]

    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.float() @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr.float()  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-4)



    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    # print(pixel_coords.reshape(b,h,w,2).shape)
    return pixel_coords.reshape(b,h,w,2)

def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    b, _, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    pixel_coords = pixel_coords.to(depth.device)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).reshape(b, 3, -1)  # [B, 3, H*W]
    #print("-"*10)
    #print("Pixel", pixel_coords.device)
    #print("Depth", depth.device)
    #print("intrinsics_inv",intrinsics_inv.device)
    #print("current_pixel_coords",current_pixel_coords.device)
    #print("-"*10)
    cam_coords = (intrinsics_inv.float() @ current_pixel_coords.float())
    cam_coords = cam_coords.reshape(b, 3, h, w)
    return cam_coords * depth.clamp(min=1e-1)


def quat2mat(quat):

    x, y, z, w = quat[:,0], quat[:,1], quat[:,2], quat[:,3]

    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    n = w2 + x2 + y2 + z2
    x = x / n
    y = y / n
    z = z / n
    w = w / n
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([1 - 2*y2 - 2*z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, 1 - 2*x2 - 2*z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, 1 - 2*x2 - 2*y2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec):

    size_list = list(vec.size())

    if len(size_list) == 3:
        # if dimension is [B 4 4] for multiview blender dataset

        return vec
    else:
        # If dimension is [B 7] for multiview nocs dataset
        b = vec.size(0)
        translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
        rot = vec[:,3:]
        rot_mat = quat2mat(rot)  # [B, 3, 3]

        invert_mat = torch.eye(4)
        invert_mat[0, 0] *= -1
        invert_mat[1, 1] *= -1

        # Adding 0.5 offset for dataset
        transform_mat = torch.cat([rot_mat,   (translation) + 0.5], dim=2)  # [B, 3, 4]
        transform_mat = torch.cat([transform_mat, torch.tensor([[0,0,0,1]]).unsqueeze(0).expand(1,1,4).type_as(transform_mat).repeat(b, 1, 1)], dim=1) # [B, 4, 4]
        return transform_mat @ invert_mat.type_as(transform_mat)

def inverse_warp(tgt_image, depth, intrinsics, src_pose, tgt_pose):

    src_camera_coords = pixel2cam(depth, intrinsics.inverse())
    src_pose_mat = pose_vec2mat(src_pose)
    tgt_pose_mat = pose_vec2mat(tgt_pose)

    src_cam_to_tgt_cam = tgt_pose_mat.inverse() @ src_pose_mat
    tgt_cam_2_proj = intrinsics @ src_cam_to_tgt_cam[:, :3, :] # Bx3x3 Bx3x4
    rot, tr = tgt_cam_2_proj[:,:,:3], tgt_cam_2_proj[:,:,-1:]
    tgt_pix_coords = cam2pixel(src_camera_coords, rot, tr)

    tgt_image = tgt_image.type_as(tgt_pix_coords)
    projected_img = F.grid_sample(tgt_image, tgt_pix_coords, padding_mode='zeros', align_corners=False)
    valid_points = tgt_pix_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points

def inverse_warp_2(tgt_image, depth, intrinsics, src_pose, tgt_pose):

    '''
    Inverse warp function using Kornia
    '''

    src_pose_mat = pose_vec2mat(src_pose)
    tgt_pose_mat = pose_vec2mat(tgt_pose)

    b = tgt_image.size(0)
    h = torch.tensor(tgt_image.size(2)).repeat(b)
    w = torch.tensor(tgt_image.size(3)).repeat(b)

    intrinsics = torch.cat([intrinsics.float(), torch.tensor([[0, 0, 0]]).unsqueeze(2).expand(1, 3, 1).type_as(intrinsics).repeat(b, 1, 1).float()], dim = 2)
    intrinsics = torch.cat([intrinsics, torch.tensor([[0, 0, 0, 1]]).expand(1, 1, 4).type_as(intrinsics).repeat(b, 1, 1).float() ], dim = 1)


    pinhole_tgt = kornia.geometry.PinholeCamera(intrinsics, tgt_pose_mat.float(), h, w)
    pinhole_src = kornia.geometry.PinholeCamera(intrinsics, src_pose_mat.float(), h, w)

    image_src = kornia.geometry.depth_warp(pinhole_tgt, pinhole_src, depth.float(), tgt_image.float(), tgt_image.size(2), tgt_image.size(3))


    return image_src, image_src

def project_depth_point_cloud(depth, intrinsics, src_pose, tgt_pose):

    '''
    Project point cloud from src to tgt pose

    '''

    src_camera_coords = pixel2cam(depth, intrinsics.inverse()) # [B, 3, H, W]
    b, _, h, w = src_camera_coords.size()
    src_pose_mat = pose_vec2mat(src_pose)
    tgt_pose_mat = pose_vec2mat(tgt_pose)

    # source camera coordinates
    src_camera_coords = src_camera_coords.reshape(b, 3, h*w)
    src_cam_to_tgt_cam = tgt_pose_mat.inverse() @ src_pose_mat
    ones = torch.ones((b, 1, h*w), device=src_camera_coords.device)
    #print("ones",ones.device)
    #print("src_camera_coords",src_camera_coords.device)
    src_camera_coords_homogeneous = torch.cat([src_camera_coords, ones], dim = 1) # [B, 4, H*W]

    # destination camera coordinates
    projected_coords = src_cam_to_tgt_cam.float() @ src_camera_coords_homogeneous.float() # [B, 4, H*W]
    projected_coords = projected_coords[:, :3, :]

    return src_camera_coords, projected_coords

def NOCS_map_2_point_cloud(nocs_image_tensor, mask):
    '''
    Convert NOCS maps to point cloud

    Input:
        nocs_image_tensor - [B, 3, H, W] - torch tensor
        mask              - [B, H, W] - torch tensor
    Returns:
        nocs_point_cloud_list   - B element list - [3, masked dims]
        indices_list            - B element list - [2, masked dims]
    '''

    indices_list = []
    nocs_point_cloud_list = []

    B, views, H, W = nocs_image_tensor.shape

    for i in range(nocs_image_tensor.shape[0]):

        ind = torch.from_numpy(((mask[i, :, :] > 0.5).nonzero().cpu()).numpy())
        h = ind[:, 0]
        w = ind[:, 1]

        #torch.sigmoid((mask[i, :, :] - 0.5)* 100)

        #h = h.detach()
        #w = w.detach()
        #print(h.max(), w.max(), h.min(), w.min())
        nocs_point_cloud = nocs_image_tensor[i, :, h, w] # [3, mask]
        nocs_point_cloud.detach_()
        nocs_point_cloud_list.append(nocs_point_cloud)
        indices_list.append(torch.stack([h, w]).detach())  # [2, mask]

    return nocs_point_cloud_list, indices_list

def get_NOCS_correspondences(nocs_image_tensor_source, mask_source, nocs_image_tensor_target, mask_target):
    '''
    Get NOCS correspondences
    Input:
        nocs_image_tensor_source - [B, 3, H, W]
        mask_source              - [B, H, W]
        nocs_image_tensor_target - [B, 3, H, W]
        mask_target              - [B, H, W]

    Returns:
        indices_depth_list       - list of tensors with indices of shape [2, masked_dim]
    '''

    B, views, H, W = nocs_image_tensor_source.shape
    indices_depth_list_target = []
    indices_depth_list_source = []

    for i in range(B):

        nocs_point_cloud_list_source, indices_list_source = NOCS_map_2_point_cloud(nocs_image_tensor_source[i, :, :, :].unsqueeze(0), mask_source[i, 0, :, :].unsqueeze(0))

        nocs_point_cloud_list_target, indices_list_target = NOCS_map_2_point_cloud(nocs_image_tensor_target[i, :, :, :].unsqueeze(0), mask_target[i, 0, :, :].unsqueeze(0))

        pc_1, ind_1 = nocs_point_cloud_list_source[0], indices_list_source[0] # [3, mask_size], [2, mask_size]
        pc_2, ind_2 = nocs_point_cloud_list_target[0], indices_list_target[0] # [3, mask_size]

        # Perform NOCS KNN matching
        out = knn_points(pc_1.transpose(0, 1).unsqueeze(0), pc_2.transpose(0, 1).unsqueeze(0)) # [1, masked_dim, 3]
        corresponding_idx = out.idx[0, :, 0] # [masked_dim]
        corresponding_idx = ind_2[:, corresponding_idx]

        indices_depth_list_source.append(ind_1)
        indices_depth_list_target.append(corresponding_idx)

    return indices_depth_list_source, indices_depth_list_target



if __name__ == "__main__":

    src_pose = torch.tensor([[1663.45703125, 46.258087158203128, -2127.346435546875, 0.008096654899418354, -0.3257482051849365, 0.0027897413820028307, 0.9454177618026733]])
    tgt_pose = torch.tensor([[1889.214599609375, 221.49795532226563, -1699.667724609375, 0.039696164429187778, -0.4065377712249756, 0.01768353208899498, 0.9125999212265015]])
    src_pose_2 = torch.tensor([[2011.62060546875, 374.8108215332031, -1255.8643798828125,0.06847226619720459, -0.48349833488464358, 0.03797297552227974, 0.8718366026878357]])


    depth = Image.open('./test-images/depth.png')
    depth = np.array(depth)
    depth = depth_decode(depth)

    depth = torch.tensor(depth).unsqueeze(0).unsqueeze(1).float()

    # print(depth)
    # plt.imshow(depth[0][0])
    # plt.show()
    tgt_image = cv2.imread('./test-images/rgb.png')
    tgt_image = torch.tensor(tgt_image).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
    intrinsics = torch.tensor([
        [617.1,0.0,320.0],
        [0.0,617.1,240.0],
        [0.0,0.0,1.0],
    ])

    scale_factor = 1
    src_pose[0, :3] = src_pose[0, :3] / scale_factor
    tgt_pose[0, :3] = tgt_pose[0, :3] / scale_factor
    src_pose_2[0, :3] = src_pose_2[0, :3] / scale_factor

    x_factor = -1
    src_pose[0, 0] = src_pose[0, 0] * x_factor
    tgt_pose[0, 0] = tgt_pose[0, 0] * x_factor
    src_pose_2[0, 0] = src_pose_2[0, 0] * x_factor

    src_pose[0, 4:6] = src_pose[0, 4:6] * -1
    tgt_pose[0, 4:6] = tgt_pose[0, 4:6] * -1
    src_pose_2[0, 4:6] = src_pose_2[0, 4:6] * -1


    intrinsics = intrinsics.unsqueeze(0)

    warp=inverse_warp(tgt_image, depth, intrinsics, tgt_pose, src_pose)
    warp=warp[0].permute(0,2,3,1)
    plt.imshow(warp[0])
    plt.show()
