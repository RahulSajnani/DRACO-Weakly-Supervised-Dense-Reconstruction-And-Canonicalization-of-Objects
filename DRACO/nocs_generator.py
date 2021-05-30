import torch, torchvision
from Data_Loaders import data_loader
from torch.utils.data import Dataset, DataLoader
from tk3dv.nocstools.aligning import estimateSimilarityUmeyama
import inv_changed
import numpy as np
import matplotlib.pyplot as plt

def find_closest_point(coords, visible_keypoints):
    '''
    Find the closest point in 3D point cloud if the key point is outside the mask
    '''
    for i in range(visible_keypoints.shape[0]):

        # Find minimum distance between keypoints on mesh and keypoints
        min_dist = torch.min(torch.mean(torch.abs(coords - visible_keypoints[i, :2, np.newaxis]), axis = 0))
        loc=torch.where(torch.mean(torch.abs(coords - visible_keypoints[i, :2, np.newaxis]), axis = 0) == min_dist)

        loc = loc[0][0]
        #print(loc)
        # Assign values back to keypoint locations
        visible_keypoints[i, 0] = coords[0, loc]
        visible_keypoints[i, 1] = coords[1, loc]

    return visible_keypoints

def generate_nocs(data_sample, depth_maps):
    '''
    Generate NOCS maps given depth maps and canonical rotation obtained from C3DPO

    Arguments:
        data_smaple - dictionary - Input dictionary given by data loader
        depth_maps - [B, views, 1, H, W] - Depth maps for the input views

    Returns:
        nocs_maps - [B, views, 1, H, W] - Generated NOCS maps

    '''
    #print(depth_maps.shape)
    cam_points = []
    nocs_maps = []
    for view in range(depth_maps.shape[1]):
        cam_points.append(inv_changed.pixel2cam(depth_maps[:, view], data_sample["intrinsics"].inverse())) # [B, 3, H, W]
        nocs_view = []
        for b in range(depth_maps.shape[0]):


            keypoint_loc = data_sample["keypoints"][b, view]
            c3dpo_cam_coords = data_sample["c3dpo_cam_coords"][b, view]
            c3dpo_rotation = data_sample["c3dpo_rotation"][b, view]
            image_frame_tensor = cam_points[view][b]
            image_mask = data_sample["masks"][b, view][0]
            h = image_frame_tensor.shape[1]
            w = image_frame_tensor.shape[2]
            image_frame_cloud = image_frame_tensor.reshape(3, h*w)
            # Extract visible keypoints
            visible_keypoint_loc = keypoint_loc[keypoint_loc[:, 2] >= 0.5]
            c3dpo_cam_coords = c3dpo_cam_coords[:, (keypoint_loc[:, 2] >= 0.5)]

            # Get locations of image coordinates that have depth values less than max depth
            coords = torch.where(image_frame_tensor[2, :, :] < depth_maps.max())
            coords = torch.stack([coords[1], coords[0]], axis = 0)

            visible_keypoints = find_closest_point(coords, visible_keypoint_loc)

            visible_keypoints = visible_keypoints.cpu().numpy()
            image_visible_kps = image_frame_tensor[ :, visible_keypoints[:, 1], visible_keypoints[:, 0]]

            # Estimate Umeyama alignment scales, rotations and translation
            Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(c3dpo_cam_coords.cpu().numpy(), image_visible_kps.detach().cpu().numpy())

            Translation = torch.from_numpy(Translation).type_as(data_sample["c3dpo_cam_coords"])
            Rotation = torch.from_numpy(Rotation).type_as(data_sample["c3dpo_cam_coords"])



            object_scaled_c3dpo = torch.from_numpy(Scales).unsqueeze(1).type_as(data_sample["c3dpo_cam_coords"]) * data_sample["c3dpo_cam_coords"][b, view]

            # Transforming the c3dpo points to canonical frame
            canonical_point_cloud_c3dpo = c3dpo_rotation.transpose(0, 1) @ object_scaled_c3dpo.float()

            # Obtain the max range along all dimensions to finally get scale
            dim_variance = torch.max(torch.abs(canonical_point_cloud_c3dpo.min(axis = 1).values - canonical_point_cloud_c3dpo.max(axis = 1).values))
            scale_point_cloud_norm = 0.85 / (dim_variance + 0.00001)

            # Transforming image point cloud to canonical frame
            image_frame_cloud = scale_point_cloud_norm * (c3dpo_rotation.transpose(0, 1) @ Rotation.T @ (image_frame_cloud.reshape(3, h*w) - Translation.unsqueeze(1).repeat(1, h*w))) + 0.5

            # image_frame_cloud.clamp(min=0, max = 1)
            image_frame_cloud = image_frame_cloud.reshape((3, h, w))

            nocs_map = torch.ones((3, h, w)).type_as(image_frame_cloud)
            image_mask = image_mask > 0.5
            nocs_map[:, image_mask ] = image_frame_cloud[:, image_mask ]
            nocs_map = nocs_map.clamp(min = 0.0, max = 1.0)
            nocs_view.append(nocs_map)

        nocs_view = torch.stack(nocs_view, dim = 0)
        nocs_maps.append(nocs_view)

    cam_points = torch.stack(cam_points, dim = 1) # [B, views, 3, H, W]
    nocs_maps = torch.stack(nocs_maps, dim = 1)

    return nocs_maps

def generate_nocs_single(data_sample, depth_map):
    '''
    Generate NOCS maps given depth maps and canonical rotation obtained from C3DPO

    Arguments:
        data_smaple - dictionary - Input dictionary given by data loader
        depth_maps - [1, H, W] - Depth maps for the input views

    Returns:
        nocs_maps - [1, H, W] - Generated NOCS maps

    '''


    cam_points = (inv_changed.pixel2cam(depth_map, data_sample["intrinsics"].inverse()))[0] # [B, 3, H, W]
    print(cam_points.shape)
    keypoint_loc = data_sample["keypoints"]
    c3dpo_cam_coords = data_sample["c3dpo_cam_coords"]
    c3dpo_rotation = data_sample["c3dpo_rotation"]

    image_frame_tensor = cam_points
    image_mask = data_sample["masks"][0]
    h = image_frame_tensor.shape[1]
    w = image_frame_tensor.shape[2]
    image_frame_cloud = image_frame_tensor.reshape(3, h*w)
    # Extract visible keypoints
    visible_keypoint_loc = keypoint_loc[keypoint_loc[:, 2] >= 0.5]
    c3dpo_cam_coords = c3dpo_cam_coords[:, (keypoint_loc[:, 2] >= 0.5)]

    # Get locations of image coordinates that have depth values less than max depth
    coords = torch.where(image_frame_tensor[2, :, :] < depth_map.max())
    coords = torch.stack([coords[1], coords[0]], axis = 0)

    visible_keypoints = find_closest_point(coords, visible_keypoint_loc)

    visible_keypoints = visible_keypoints.cpu().numpy()
    image_visible_kps = image_frame_tensor[ :, visible_keypoints[:, 1], visible_keypoints[:, 0]]

    # Estimate Umeyama alignment scales, rotations and translation
    Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(c3dpo_cam_coords.cpu().numpy(), image_visible_kps.detach().cpu().numpy())

    Translation = torch.from_numpy(Translation).type_as(data_sample["c3dpo_cam_coords"])
    Rotation = torch.from_numpy(Rotation).type_as(data_sample["c3dpo_cam_coords"])



    object_scaled_c3dpo = torch.from_numpy(Scales).unsqueeze(1).type_as(data_sample["c3dpo_cam_coords"]) * data_sample["c3dpo_cam_coords"]

    # Transforming the c3dpo points to canonical frame
    canonical_point_cloud_c3dpo = c3dpo_rotation.transpose(0, 1) @ object_scaled_c3dpo.float()

    # Obtain the max range along all dimensions to finally get scale
    dim_variance = torch.max(torch.abs(canonical_point_cloud_c3dpo.min(axis = 1).values - canonical_point_cloud_c3dpo.max(axis = 1).values))
    scale_point_cloud_norm = 0.85 / (dim_variance + 0.00001)

    # Transforming image point cloud to canonical frame
    image_frame_cloud = scale_point_cloud_norm * (c3dpo_rotation.transpose(0, 1) @ Rotation.T @ (image_frame_cloud.reshape(3, h*w) - Translation.unsqueeze(1).repeat(1, h*w))) + 0.5

    # image_frame_cloud.clamp(min=0, max = 1)
    image_frame_cloud = image_frame_cloud.reshape((3, h, w))

    nocs_map = torch.ones((3, h, w)).type_as(image_frame_cloud)
    image_mask = image_mask > 0.5
    nocs_map[:, image_mask ] = image_frame_cloud[:, image_mask ]
    nocs_map = nocs_map.clamp(min = 0.0, max = 1.0)



    return nocs_map



if __name__ == "__main__":

    dataset_path = "../data/canonical_dataset"
    dataset = data_loader.MultiView_canonical_dataset_blender(dataset_path=dataset_path, train = 1, num_views = 3, gt_depth = True, normalize = False, jitter = False, canonical=True)

    data = dataset[0]


    dataloader = DataLoader(dataset, batch_size = 2, shuffle=True)
    data = list(enumerate(dataloader))
    # print(data[0])
    data_sample = data[0][1]
    # print(data_sample.keys())

    nocs = generate_nocs(data_sample, data_sample["depths"])
    print(torch.max(nocs))
    nocs = nocs.numpy()
    print(nocs.shape)

    # h, w = nocs.shape[-2:]
    plt.subplot(221)
    plt.imshow(nocs[0, 0].transpose((1, 2, 0)))

    plt.subplot(222)
    plt.imshow(nocs[0, 1].transpose((1, 2, 0)))

    plt.subplot(223)
    plt.imshow(nocs[0, 2].transpose((1, 2, 0)))

    plt.subplot(224)
    plt.imshow(nocs[1, 1].transpose((1, 2, 0)))


    plt.show()

