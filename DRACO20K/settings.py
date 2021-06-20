"""
Adapted from https://github.com/weiaicunzai/blender_shapenet_render
"""

g_shapenet_path = '../../shapenet_categories/'
g_output_path = '../generated_data'
g_background_image_path = "../unlabeled2017"
g_number_images = 50
g_max_models = 200

# Cars 3D key points json obtained from here https://github.com/qq456cvb/KeypointNet.git
####################################################### Uncomment for the required dataset

##### CARS
g_kp_json = "./keypoints_json/car.json"
g_gt_and_equal = False
g_keypoint_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11,
                   12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19, 20:20, 21:21}
g_num_kps = 22


##### AIRPLANE
# g_kp_json = "./keypoints_json/airplane.json"
# g_gt_and_equal = True
# g_keypoint_dict = {0:0, 1:1, 3:2, 5:3, 7:4, 8:5, 12:6, 13:7, 14:8, 15:9}
# g_keypoint_dict = {0:0, 3:1, 5:2, 7:3, 8:4, 12:5, 13:6, 14:7}

# g_num_kps = 14


##### CHAIR
# g_kp_json = "./keypoints_json/chair.json"
# g_keypoint_dict = {0: 0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 8:7}
# g_gt_and_equal = False
# g_num_kps = 14


##### LAPTOPS
#g_kp_json = "./keypoints_json/laptop.json"
#g_gt_and_equal = False
#g_keypoint_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
#g_num_kps = 6


##### MUGS
#g_kp_json = "./keypoints_json/mug.json"
#g_gt_and_equal = False
#g_keypoint_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10}
#g_num_kps = 11

#if you have multiple viewpoint files, add to the dict
#files contains azimuth,elevation,tilt angles and distance for each row
g_view_point_file ={
    'chair' : 'view_points/chair.txt',
    'bottle' : 'view_points/bottle.txt',
    'table' : 'view_points/diningtable.txt',
    'sofa' : 'view_points/sofa.txt',
    'bed' : 'view_points/bed.txt'
}

g_render_objs = ['chair', 'table', 'sofa', 'bed', 'bottle']

#background image composite
#enum in [‘RELATIVE’, ‘ABSOLUTE’, ‘SCENE_SIZE’, ‘RENDER_SIZE’], default ‘RELATIVE’
g_scale_space = 'RENDER_SIZE'
g_use_film_transparent = True

#camera:
#enum in [‘QUATERNION’, ‘XYZ’, ‘XZY’, ‘YXZ’, ‘YZX’, ‘ZXY’, ‘ZYX’, ‘AXIS_ANGLE’]
g_rotation_mode = 'QUATERNION'
g_depth_clip_start = 0.5
g_depth_clip_end = 10

#output:

#enum in [‘BW’, ‘RGB’, ‘RGBA’], default ‘BW’
g_rgb_color_mode = 'RGB'
#enum in [‘8’, ‘10’, ‘12’, ‘16’, ‘32’], default ‘8’
g_rgb_color_depth = '16'
g_rgb_file_format = 'PNG'

g_depth_color_mode = 'BW'
g_depth_color_depth = '8'
g_depth_file_format = 'PNG'

g_depth_use_overwrite = True
g_depth_use_file_extension = True

#dimension:

#engine type [CYCLES, BLENDER_RENDER]
g_engine_type = 'CYCLES'

#output image size =  (g_resolution_x * resolution_percentage%, g_resolution_y * resolution_percentage%)
g_resolution_x = 640
g_resolution_y = 480
g_resolution_percentage = 100


#performance:

g_gpu_render_enable = False

#if you are using gpu render, recommand to set hilbert spiral to 256 or 512
#default value for cpu render is fine
g_hilbert_spiral = 256

#total 55 categories
g_shapenet_categlory_pair = {
    'table' : '04379243',
    'jar' : '03593526',
    'skateboard' : '04225987',
    'car' : '02958343',
    'bottle' : '02876657',
    'tower' : '04460130',
    'chair' : '03001627',
    'bookshelf' : '02871439',
    'camera' : '02942699',
    'airplane' : '02691156',
    'laptop' : '03642806',
    'basket' : '02801938',
    'sofa' : '04256520',
    'knife' : '03624134',
    'can' : '02946921',
    'rifle' : '04090263',
    'train' : '04468005',
    'pillow' : '03938244',
    'lamp' : '03636649',
    'trash bin' : '02747177',
    'mailbox' : '03710193',
    'watercraft' : '04530566',
    'motorbike' : '03790512',
    'dishwasher' : '03207941',
    'bench' : '02828884',
    'pistol' : '03948459',
    'rocket' : '04099429',
    'loudspeaker' : '03691459',
    'file cabinet' : '03337140',
    'bag' : '02773838',
    'cabinet' : '02933112',
    'bed' : '02818832',
    'birdhouse' : '02843684',
    'display' : '03211117',
    'piano' : '03928116',
    'earphone' : '03261776',
    'telephone' : '04401088',
    'stove' : '04330267',
    'microphone' : '03759954',
    'bus' : '02924116',
    'mug' : '03797390',
    'remote' : '04074963',
    'bathtub' : '02808440',
    'bowl' : '02880940',
    'keyboard' : '03085013',
    'guitar' : '03467517',
    'washer' : '04554684',
    'bicycle' : '02834778',
    'faucet' : '03325088',
    'printer' : '04004475',
    'cap' : '02954340',
    'clock' : '03046257',
    'helmet' : '03513137',
    'flowerpot' : '03991062',
    'microwaves' : '03761084'
}

