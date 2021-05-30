import open3d as o3d
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

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
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

def visualize_nocs_map(nocs_map, nm, image = None):
    '''
    Plots 3D point cloud from nocs map
    Arguments:
        nocs_map - [H x W x 3] - NOCS map for image
    
    Returns: 
        None
    '''
    
    h, w = nocs_map.shape[:2]
    nocs_mask = generate_mask_NOCS(nm)
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

    # vis_obj.draw_geometries([pcd, line_set])


    return pcd
    
if __name__ == "__main__":

    vis_obj = o3d.visualization
    nocs_image_path = sys.argv[1]
    image = None

    nocs_image_path_2 = sys.argv[2]

    # R = np.asarray(
    #     [
    #         [ 0.13113765, -0.96770999, -0.15196208],
    #         [-0.97693124, -0.11753038, -0.09476062],
    #         [ 0.07573157,  0.16221497, -0.97838511],
    #     ]
    # )

    # # Read the NOCS maps and construct point clouds
    nocs_map = read_image(nocs_image_path)
    pcd = visualize_nocs_map(nocs_map, nocs_map, image)
    # T = R @ (np.asarray(pcd.points) - 0.5).T + 0.5     
    nocs_map_2 = read_image(nocs_image_path_2)
    pcd_2 = visualize_nocs_map(nocs_map_2, nocs_map,image)

    pcd.paint_uniform_color([0, 0.651, 0.929])
    pcd_2.paint_uniform_color([1, 0.706, 0])

    # Visualize the point clouds
    vis_obj.draw_geometries([pcd, pcd_2])

    # # Center the two NOCS point clouds 
    # pcd_points = np.asarray(pcd.points) - 0.5
    # pcd_points_2 = np.asarray(pcd_2.points) - 0.5
    # pcd.points = o3d.utility.Vector3dVector(pcd_points[:,:3])
    # pcd_2.points = o3d.utility.Vector3dVector(pcd_points_2[:,:3])

    # # Visualize the centered point clouds
    # vis_obj.draw_geometries([pcd, pcd_2])
    # # print(pcd_points.shape)
    # Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(pcd_points.T, pcd_points_2.T)

    # vis_obj.draw_geometries([pcd, pcd_2])

    # reg_p2p = o3d.registration.registration_icp(
    #     pcd, pcd_2, 0.2, OutTransform,
    #     o3d.registration.TransformationEstimationPointToPoint())

    # print(reg_p2p.transformation)

    

    # draw_registration_result(pcd, pcd_2, reg_p2p.transformation)




    # gt_files = sorted(glob.glob("../../../../test/*/*_C3DPO_00.png"))
    # pred_files = sorted(glob.glob("../../../../nocs_outputs/pipeline1/*/*_nocs.jpg"))

    # chamfer_forward = []
    # chamfer_backward = []

    # photometric = []

    # for i in tqdm(range(len(gt_files))):
    #     gt_c3dpo = gt_files[i]
    #     pred_c3dpo = pred_files[i]

    #     c3dpo_map = read_image(gt_c3dpo)
    #     pred_map = read_image(pred_c3dpo)

    #     pcd_c3dpo = visualize_nocs_map(c3dpo_map, c3dpo_map)
    #     pcd_pred  = visualize_nocs_map(pred_map, c3dpo_map)

    #     c3dpo_points = np.asarray(pcd_c3dpo.points)
    #     pred_points = np.asarray(pcd_pred.points)

    #     c3dpo_points = torch.from_numpy(c3dpo_points.astype(float)).unsqueeze(0)
    #     pred_points = torch.from_numpy(pred_points.astype(float)).unsqueeze(0)

    #     # draw_registration_result(pcd_c3dpo, pcd_pred, np.eye(4))

    #     c1, _ = chamfer_distance(c3dpo_points.float(), pred_points.float())
    #     c2, _ = chamfer_distance(pred_points.float(), c3dpo_points.float())

    #     chamfer_forward.append(c1.item())
    #     chamfer_backward.append(c2.item())

    #     p = photometric_error(gt=c3dpo_map , pred=pred_map)
    #     photometric.append(p)


    # photometric = np.array(photometric)
    # chamfer_forward = np.array(chamfer_forward)
    # chamfer_backward = np.array(chamfer_backward)
    
    # np.save('photometric.npy', photometric)
    # np.save('chamfer_forward.npy', chamfer_forward)
    # np.save('chamfer_backward.npy', chamfer_backward)

    # print("Photometric:",np.mean(photometric))
    # print("Chamfer Forward:",np.mean(chamfer_forward))
    # print("Chamfer Backward:",np.mean(chamfer_backward))





















    # print(len(gt_files))
    image = None
    arr = []
    x = []
    y = []
    z = []

    gt_files = sorted(glob.glob("/home/aadilmehdi/RRC/WSNOCS/14102020/planes/ValSet/val/02691156/*"))

    for i in tqdm(range(len(gt_files))):
        gt_c3dpo = sorted(glob.glob( gt_files[i] + '/*_NOXRayTL_00.png'))
        gt_nocs  = sorted(glob.glob( gt_files[i] + '/*_C3DPO_00.png'))
        for j in range(len(gt_c3dpo)):
            nocs_image_gt = gt_c3dpo[j]
            c3dpo_image_gt = gt_nocs[j]

            nocs_gt = read_image(nocs_image_gt)
            c3dpo_gt = read_image(c3dpo_image_gt)

            pcd_nocs = visualize_nocs_map(nocs_gt, nocs_gt, image)
            pcd_c3dpo = visualize_nocs_map(c3dpo_gt, nocs_gt, image)


            draw_registration_result(pcd_nocs, pcd_c3dpo, np.eye(4))

            nocs_points = np.asarray(pcd_nocs.points) - 0.5
            c3dpo_points = np.asarray(pcd_c3dpo.points) - 0.5
            pcd_nocs.points = o3d.utility.Vector3dVector(nocs_points[:,:3])
            pcd_c3dpo.points = o3d.utility.Vector3dVector(c3dpo_points[:,:3])

            Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(nocs_points.T, c3dpo_points.T)

            reg_p2p = o3d.registration.registration_icp(
                pcd_nocs, pcd_c3dpo, 0.2, OutTransform,
                o3d.registration.TransformationEstimationPointToPoint())


            arr.append(reg_p2p.transformation)
            arr2 = np.array(arr)

            # print(reg_p2p.transformation)
            mat_loc = mathutils.Matrix(reg_p2p.transformation)


            print(reg_p2p.transformation)
            # print("Variance", np.var(arr2))
            # print("Mean", np.mean(arr2, axis=0))
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # print()
            # eul = mat_loc.to_euler()
            print(mat_loc)
            # x.append(math.degrees(eul.x))
            # y.append(math.degrees(eul.y))
            # z.append(math.degrees(eul.z))
            # print('x:',math.degrees(eul.x))
            # print('y:',math.degrees(eul.y))
            # print('z:',math.degrees(eul.z))
            # print()
            draw_registration_result(pcd_nocs, pcd_c3dpo, reg_p2p.transformation)

    # x = np.array(x)
    # np.save('x.npy', x)
    # y = np.array(y)
    # np.save('y.npy', y)
    # z = np.array(z)
    # np.save('z.npy', z)


    # print("Mean X (Degrees)",np.mean(x))
    # print("Mean Y (Degrees)",np.mean(y))
    # print("Mean Z (Degrees)",np.mean(z))

    # print()

    # print("Var X (Degrees)",np.var(x))
    # print("Var Y (Degrees)",np.var(y))
    # print("Var Z (Degrees)",np.var(z))

    # print()
    # print("X")
    # for i in x:
    #     print(i)
    # print()
    # print("Y")
    # for i in y:
    #     print(i)
    # print()
    # print("Z")
    # for i in z:
    #     print(i)

