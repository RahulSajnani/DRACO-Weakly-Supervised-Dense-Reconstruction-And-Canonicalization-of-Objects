import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from PIL import Image
import json
from mpl_toolkits.mplot3d import Axes3D as mpl_3D
#import pptk
import glob
import os
import tk3dv
import argparse

global scale_factor


FLOAT_EPS = np.finfo(np.float).eps
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
    image_points = np.stack([x_i, y_i, np.ones_like(x_i)], axis = -1)
    coords = image_points.reshape(x*y, 3)
    # coords = np.indices((x, y)).reshape(2, -1)
    # return np.vstack((coords, np.ones(coords.shape[1])))
    #print(coords)
    return coords.T, image_points

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

    depth = scale_factor * (1.0 - depth)
    #depth = 8 - 8*depth
    #depth = 1/(3*depth + 0.001)
    depth = depth.reshape(1, -1)
    return depth

def depth_2_point_cloud(invK, image, depth_image, depth_tiff=None):
    '''
    Convert depth map to point cloud

    '''
    points_hom, image_coords = get_grid(image.shape[0], image.shape[1])


    print("Image coords: ", image_coords.transpose(2, 0, 1)[:, np.newaxis, :, :].shape)
    image_coords_tensor = torch.from_numpy(image_coords.transpose(2, 0, 1))
    image_frame_cloud =  torch.from_numpy(invK).float().unsqueeze(0) @ image_coords_tensor.float().unsqueeze(0).reshape(1, 3, -1)
    print(image_frame_cloud.size())
    # image_frame_cloud =  invK @ image_coords.transpose(2, 0, 1)[:, np.newaxis, :, :]

    if depth_tiff != None:
        print('tiff\n')
        depth = extract_depth_tiff(depth_tiff)
    else:
        depth = extract_depth(depth_image)
    # depth_map, depth = depth_decode(depth_image)

    image_frame_cloud = (image_frame_cloud / image_frame_cloud[:, 2, :]) * torch.from_numpy(depth).clamp(max = 10)

    print(np.min(depth), np.max(depth[depth<30]))
    point_3D = invK @ points_hom
    point_3D = point_3D / point_3D[2, :]
    point_3D = point_3D * depth

    return point_3D, image_frame_cloud

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
    #print(invK)
    point_cloud, image_frame_cloud_tensor = depth_2_point_cloud(invK, image, depth, depth_tiff)
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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.T)
    pcd.colors = o3d.utility.Vector3dVector(image_colors)
    # print(pcd.colors)
    ply_name = "%s/frame_%06d_point_cloud.ply" % (output_directory, num)
    # o3d.io.write_point_cloud(ply_name, pcd)
    return point_cloud, pcd, image_frame_cloud_tensor, mask

def extract_depth(depth_map):

    '''
    Get a depth vector from image
    '''

    depth = depth_map.reshape(1, -1)
    depth = scale_factor*(1.0 - depth)
    #depth = 8 - 8 * depth
    ###############################
    #depth = 1 / (depth + 0.001)

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

def scale_point_cloud_umeyama(image_frame_tensor, cam_coords_canonical, keypoint_locs, mask, image):
    '''
    Umeyama alignment
    '''

    # image_frame_tensor[:, :, mask[0]]
    mask_image = mask.copy()
    points_hom, image_coords = get_grid(mask_image.shape[0], mask_image.shape[1])
    # print(image_coords.shape, points_hom.shape)

    mask = torch.from_numpy(mask).reshape(1, -1)

    # print(keypoint_locs.shape)
    # print(mask.size())

    image_frame_tensor = image_frame_tensor.reshape(1, 3, image.shape[0], image.shape[1])
    visible_keypoints = keypoint_locs[keypoint_locs[:, 2] == 1]

    # print(np.where(mask_image[image_coords[0], image_coords[1]] >= 0.5))
    # print(image_frame_tensor.size())

    coords = torch.where(image_frame_tensor[0, 2, :, :] < 10)
    coords = torch.stack([coords[1], coords[0]], axis = 0)
   # print(coords)
    if coords.shape[1] > 0:
        for i in range(visible_keypoints.shape[0]):


            min_dist = torch.min(torch.mean(torch.abs(coords - visible_keypoints[i, :2, np.newaxis]), axis = 0))
            loc=torch.where(torch.mean(torch.abs(coords - visible_keypoints[i, :2, np.newaxis]), axis = 0) == min_dist)
            #print(loc)
            loc = loc[0][0]
            #print(loc)
            visible_keypoints[i, 0] = coords[0, loc]
            visible_keypoints[i, 1] = coords[1, loc]

    image_visible_kps_numpy = image_frame_tensor[:, :, visible_keypoints[:, 1], visible_keypoints[:, 0]][0].numpy()
    visible_cam_coords_c3dpo = cam_coords_canonical[:, :, (keypoint_locs[:, 2] == 1)][0]

    # print(image_visible_kps_numpy[2, :])
    # points_bool_vector = image_visible_kps_numpy[2, :] < (np.mean(image_visible_kps_numpy[2, :]))

    # print(points_bool_vector)
    # image_visible_kps_numpy = image_visible_kps_numpy[:, points_bool_vector]
    # visible_cam_coords_c3dpo = visible_cam_coords_c3dpo[:, points_bool_vector]

    Scales, Rotation, Translation, OutTransform = tk3dv.nocstools.aligning.estimateSimilarityUmeyama(visible_cam_coords_c3dpo, image_visible_kps_numpy)


    object_scaled_c3dpo = cam_coords_canonical[0] * Scales[0]

    # print("Scale\n", Scales)
    # print("Rotation\n", Rotation)
    # print("Translation\n", Translation)
    # print("Transformation\n", OutTransform)

    return torch.tensor(np.array([Translation])), object_scaled_c3dpo, torch.from_numpy(np.array([Rotation]))

def scale_point_cloud(image_frame_tensor, cam_coords_canonical, keypoint_locs, mask, image):
    '''
    Find scale to match keypoint cam_coords with point cloud
    '''

    # image_frame_tensor[:, :, mask[0]]

    # print(keypoint_locs.shape)
    #print(mask.size())
    # print(keypoint_locs.shape)
    image_frame_tensor = image_frame_tensor.reshape(1, 3, image.shape[0], image.shape[1])
    visible_keypoints = keypoint_locs[keypoint_locs[:, 2] == 1]
    # print(image_frame_tensor.size())
    image_visible_kps_tensor = image_frame_tensor[:, :, visible_keypoints[:, 1], visible_keypoints[:, 0]]

    visible_cam_coords_c3dpo = cam_coords_canonical[:, :, (keypoint_locs[:, 2] == 1)]

    values, indices = torch.topk(image_visible_kps_tensor[:, 2, :], k = 2, largest = False)

    # print(indices, values)

    scale_c3dpo = np.sqrt(((visible_cam_coords_c3dpo[:, :, indices[0,0]] - visible_cam_coords_c3dpo[:, :, indices[0,1]])**2).sum())
    scale_point_cloud = torch.sqrt((image_visible_kps_tensor[:, :, indices[0,0]] - image_visible_kps_tensor[:, :, indices[0,1]])**2).sum()

    visible_cam_coords_c3dpo = visible_cam_coords_c3dpo[:, :, (image_visible_kps_tensor[:, 2, :] < image_visible_kps_tensor[:, 2, :].mean()).squeeze()]
    image_visible_kps_tensor = image_visible_kps_tensor[:, :,(image_visible_kps_tensor[:, 2, :] < image_visible_kps_tensor[:, 2, :].mean()).squeeze()]

    # image_visible_kps_tensor_temp = image_visible_kps_tensor[:, :, indices[0]]

    # print(image_visible_kps_tensor.size())

    scale_factor = scale_point_cloud / (scale_c3dpo + 0.001)
    #print(scale_factor)

    visible_cam_coords_c3dpo *= scale_factor.numpy()

    # print(cam_coords_canonical.sha)
    object_scaled_c3dpo = cam_coords_canonical[0] * scale_factor.numpy()

    # extremums =

    translation = (image_visible_kps_tensor - torch.from_numpy(visible_cam_coords_c3dpo)).mean(axis = 2)

    # print(translation)
    # print(visible_cam_coords_c3dpo)
    # print(image_visible_kps_tensor)
    return translation, object_scaled_c3dpo

if __name__ == "__main__":

    ################################# Argument Parser #####################################

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", help = "Path to images", required=True)
    parser.add_argument("--output", help = "Output directory path", required=True)

    parser.add_argument("--depth_scale", help = "sigmoid 2 depth scale", type=int, default=10)

    args = parser.parse_args()

    #######################################################################################

    scale_factor = args.depth_scale
    # input_root = args.path
    input_directory_name = args.path
    output_directory_name = args.output

    # input_directory_name = input_directory_name

    list_image = glob.glob(input_directory_name + "*image.jpg")
    list_image.sort()
    # list_image = list_image[1200:]
    canonical_folder = args.path

    for files in list_image:

        i = files.split('/')[-1].split('_')[1]
        i = int(i)

        image_number = i
        image_name = 'frame_%06d_image.jpg' % image_number
        image_depth = 'frame_%06d_depth.jpg' % image_number
        image_mask = 'frame_%06d_mask.jpg' % image_number
        image_depth_tiff = 'frame_%06d_depth.tiff' % image_number

        canonicalization_rot = canonical_folder + "frame_%06d_rotation.npy" % image_number
        canonicalization_trans = canonical_folder + "frame_%06d_translation.npy" % image_number
        keypoints = canonical_folder + "frame_%06d_KeyPoints.npy" % image_number
        cam_canonical_coords = canonical_folder + "frame_%06d_cam_coords.npy" % image_number

        rotation_mat = np.linalg.inv(np.load(canonicalization_rot)[0])
        keypoints = np.load(keypoints)
        cam_canonical_coords = np.load(cam_canonical_coords)

        # print(cam_canonical_coords.min())
        # print(cam_canonical_coords.shape)


        # K = camera_matrix(617.1, 617.1, 315, 242)
        K = camera_matrix(888.88, 1000.0, 320, 240)
        # K = camera_matrix(571, 319.5, 239.5)

        mask = cv2.imread(input_directory_name + image_mask,  cv2.IMREAD_GRAYSCALE) / 255.0
        image = cv2.imread(input_directory_name + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        depth = cv2.imread(input_directory_name + image_depth,  cv2.IMREAD_GRAYSCALE) / 255.0
        depth_tiff = Image.open(input_directory_name + image_depth_tiff)

        try:
            # SAVE POINT CLOUD
            point_cloud, pcd, image_frame_cloud_tensor, mask = save_point_cloud(K,image, mask, depth, image_number, output_directory_name, depth_tiff)

            mask_tensor = torch.from_numpy(mask).reshape(1, -1)
            # print(image_frame_cloud_tensor[:, :, mask_tensor[0]].mean(axis = 2))

            translation, object_scaled_c3dpo, rotation_umeyama = scale_point_cloud_umeyama(image_frame_cloud_tensor, cam_canonical_coords, keypoints, mask, image)
            # translation, object_scaled_c3dpo = scale_point_cloud(image_frame_cloud_tensor, cam_canonical_coords, keypoints, mask_tensor, image)
            # print(translation, "\n", object_scaled_c3dpo)
            # print(translation_u, "\n", object_scaled_c3dpo_u)

            # print(rotation_umeyama.size(), image_frame_cloud_tensor.size())

            image_frame_cloud_tensor =  (image_frame_cloud_tensor - translation.unsqueeze(2).repeat(1, 1, image_frame_cloud_tensor.size(2)))


            # print(np.mean(point_cloud, axis = 1, keepdims = True))
            norm_point_cloud = point_cloud - translation.numpy()[0][:, np.newaxis].repeat(point_cloud.shape[1], axis=1)
            # plt.subplot(121)
            # plt.imshow(nocs_image)
            # # plt.show()
            # plt.subplot(122)
            # plt.imshow(image)
            # plt.show()

            canonical_point_cloud_c3dpo = rotation_mat @ object_scaled_c3dpo

            # print(canonical_point_cloud_c3dpo)
            dim_variance = np.max(np.abs(canonical_point_cloud_c3dpo.min(axis = 1) - canonical_point_cloud_c3dpo.max(axis = 1)))
            scale_point_cloud_norm = 1 / (dim_variance + 0.00001)
            # print(scale_point_cloud_norm)
            canonical_point_cloud = rotation_mat @ (scale_point_cloud_norm * norm_point_cloud * 0.8) + 0.5

            mask_tensor = mask_tensor * (image_frame_cloud_tensor[:, 2, :] < 10)
            image_frame_cloud_tensor[:, :, mask_tensor[0]] = (torch.from_numpy(rotation_mat).unsqueeze(0) @ (image_frame_cloud_tensor[:, :, mask_tensor[0]] * scale_point_cloud_norm * 0.8) + 0.5).clamp(min = 0, max = 1)

            image_frame_cloud_tensor[:, :, mask_tensor[0] == False] = 1.0
            canonical_tensor = image_frame_cloud_tensor.clamp(min = 0, max = 1)
            nocs_image = canonical_tensor.reshape(3, 480, 640).permute(1, 2, 0).numpy()
            canonical_point_cloud = canonical_point_cloud.clip(0, 1)

            #print(canonical_point_cloud.max(axis = 1))
            pcd.points = o3d.utility.Vector3dVector((canonical_point_cloud - 0.5).T)

            # o3d.visualization.draw_geometries([pcd])
            ply_name = "%s/frame_%06d_canonical_point_cloud.ply" % (output_directory_name, image_number)
            nocs_name = "%s/frame_%08d_nocs.png" % (output_directory_name, image_number)
            plt.imsave(nocs_name, nocs_image)
            o3d.io.write_point_cloud(ply_name, pcd)

            ply_name_nocs = "%s/frame_%06d_nocs_point_cloud.ply" % (output_directory_name, image_number)
            pcd.points = o3d.utility.Vector3dVector((canonical_point_cloud).T)
            o3d.io.write_point_cloud(ply_name_nocs, pcd)
        except:
            continue
        # break
