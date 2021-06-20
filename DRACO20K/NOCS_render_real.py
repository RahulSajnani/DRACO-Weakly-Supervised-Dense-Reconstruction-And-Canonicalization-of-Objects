'''
Adapted from https://github.com/weiaicunzai/blender_shapenet_render

Author: Rahul Sajnani
'''
import sys
import os
import random
import pickle
import bpy
import mathutils
import numpy as np
import json
import glob
from mathutils import Matrix, Vector
import bpy_extras
import imageio
abs_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(abs_path))


from settings import *
import settings

def clear_mesh():
    """
    Clears mesh in scene and deletes from .blend
    """

    # Delete mesh
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete mesh from .blend file
    for block in bpy.data.meshes:
        if block.users == 0:
                bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

def scene_setting_init(use_gpu):
    """
    initialize blender setting configurations

    """
    sce = bpy.context.scene.name
    bpy.data.scenes[sce].render.engine = g_engine_type
    bpy.data.scenes[sce].cycles.film_transparent = True
    #output
    # bpy.data.scenes[sce].render.image_settings.color_mode = g_rgb_color_mode
    # bpy.data.scenes[sce].render.image_settings.color_depth = g_rgb_color_depth
    # bpy.data.scenes[sce].render.image_settings.file_format = g_rgb_file_format

    #dimensions
    bpy.data.scenes[sce].render.resolution_x = g_resolution_x
    bpy.data.scenes[sce].render.resolution_y = g_resolution_y
    bpy.data.scenes[sce].render.resolution_percentage = g_resolution_percentage

    if use_gpu:
        bpy.data.scenes[sce].render.engine = 'CYCLES' #only cycles engine can use gpu
        bpy.data.scenes[sce].render.tile_x = g_hilbert_spiral
        bpy.data.scenes[sce].render.tile_x = g_hilbert_spiral
        bpy.data.scenes[sce].cycles.device = 'GPU'

        scene = bpy.context.scene
        scene.cycles.device = 'GPU'

def node_setting_init():
    """
    node settings for render rgb images

    mainly for compositing the background images

    https://blender.stackexchange.com/questions/180355/change-saturation-of-rendered-object-without-changing-the-background-image
    """

    bpy.context.scene.render.use_compositing = True

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    # Creating nodes ###########################################
    image_node = tree.nodes.new('CompositorNodeImage')
    scale_node = tree.nodes.new('CompositorNodeScale')
    alpha_over_node = tree.nodes.new('CompositorNodeAlphaOver')
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    file_output_node = tree.nodes.new('CompositorNodeOutputFile')

    # Nodes to fix saturation
    mask_node = tree.nodes.new('CompositorNodeIDMask')
    saturation_node = tree.nodes.new('CompositorNodeHueSat')
    alpha_over_node_2 = tree.nodes.new('CompositorNodeAlphaOver')

    nodes = {}
    # Nodes for mask and depth
    file_output_mask_node = tree.nodes.new('CompositorNodeOutputFile')
    file_output_depth_node = tree.nodes.new('CompositorNodeOutputFile')

    file_output_mask_node.format.color_mode = 'BW'
    file_output_depth_node.format.color_mode = 'BW'
    file_output_depth_node.format.color_depth = '16'


    scale_node.space = g_scale_space


    # Linking nodes #############################################
    links.new(image_node.outputs[0], scale_node.inputs[0])
    links.new(scale_node.outputs[0], alpha_over_node.inputs[1])
    links.new(alpha_over_node.outputs[0], file_output_node.inputs[0])

    # saturation fix

    links.new(render_layer_node.outputs[0], saturation_node.inputs[0])
    links.new(render_layer_node.outputs[0], alpha_over_node_2.inputs[1])
    links.new(render_layer_node.outputs[3], mask_node.inputs[0])

    links.new(mask_node.outputs[0], alpha_over_node_2.inputs[0])
    links.new(saturation_node.outputs[0], alpha_over_node_2.inputs[2])
    links.new(alpha_over_node_2.outputs[0], alpha_over_node.inputs[2])

    # Depth and Mask links

    links.new(render_layer_node.outputs[2], file_output_depth_node.inputs[0])
    # links.new(render_layer_node.outputs[1], file_output_mask_node.inputs[0])
    links.new(mask_node.outputs[0], file_output_mask_node.inputs[0])


    # Setting values for nodes####################################
    file_output_depth_node.format.file_format = 'OPEN_EXR'
    file_output_node.file_slots[0].path = 'frame_########_Color_00.png' # blender placeholder #
    file_output_depth_node.file_slots[0].path = 'frame_########_Depth_00.exr'
    file_output_mask_node.file_slots[0].path = 'frame_########_Mask_00.png'

    mask_node.index = 1
    # hue
    saturation_node.inputs[1].default_value = 0.5
    # saturation
    saturation_node.inputs[2].default_value = 1.3
    # value factor
    saturation_node.inputs[3].default_value = 2.0
    saturation_node.inputs[4].default_value = 1.0



def init_all():
    """
    init everything we need for rendering an image
    """
    scene_setting_init(g_gpu_render_enable)
    set_rendering_settings()
    node_setting_init()
    cam_obj = bpy.data.objects['Camera']
    cam_obj.rotation_mode = g_rotation_mode

    bpy.data.objects['Light'].data.energy = 0

    clear_mesh()
    add_light()

    cam_obj = bpy.data.objects['Camera']
    cam_obj.data.sensor_fit = 'VERTICAL'

    # setting camera parameters for focal length ~= 617.1
    cam_obj.data.lens = 50.0
    cam_obj.data.sensor_width = 36.0
    cam_obj.data.clip_start = 0.01
    cam_obj.data.clip_end = 30




def set_image_path(new_path):
    """
    set image output path to new_path

    Args:
        new rendered image output path
    """
    file_output_node = bpy.context.scene.node_tree.nodes[4]
    file_output_node.base_path = new_path

    file_depth_node = bpy.context.scene.node_tree.nodes[9]
    file_depth_node.base_path = new_path

    file_mask_node = bpy.context.scene.node_tree.nodes[8]
    file_mask_node.base_path = new_path


def camera_look_at_object(object_cam, object_target):
    '''
    Sets the camera quaternion automatically to look at shapenet object
    '''
    direction = object_target.location - object_cam.location

    rot_quaternion = direction.to_track_quat('-Z', 'Y')
    object_cam.rotation_quaternion = rot_quaternion


def add_light():
    '''
    Add scene lighting
    '''

    light_intensity = 15000
    # create light datablock, set attributes
    light_data = bpy.data.lights.new(name="light_1", type='SPOT')
    light_data.energy = light_intensity


    # create new object with our light datablock
    light_object = bpy.data.objects.new(name="light_1", object_data=light_data)

    light_object.rotation_mode = "QUATERNION"

    # link light object
    bpy.context.collection.objects.link(light_object)

    # make it active
    bpy.context.view_layer.objects.active = light_object
    # light_object.data.cycles.cast_shadow = False

    #change location
    light_object.location = (0, 0, 10)
    direction = mathutils.Vector((0.0,0.0, 0.0)) - light_object.location
    light_object.rotation_quaternion = direction.to_track_quat('-Z', 'Y')

    # dg = bpy.context.evaluated_depsgraph_get()
    # dg.update()

    light_data_2 = bpy.data.lights.new(name="light_2", type='SPOT')
    light_data_2.energy = light_intensity

    # create new object with our light datablock
    light_object_2 = bpy.data.objects.new(name="light_2", object_data=light_data_2)

    # link light object
    bpy.context.collection.objects.link(light_object_2)

    # make it active
    bpy.context.view_layer.objects.active = light_object_2

    #change location
    light_object_2.location = (0, 0, -10)
    # Look at origin
    direction_2 = mathutils.Vector((0.0, 0.0, 0.0)) - light_object_2.location

    print("Lights")
    print(light_object_2.rotation_quaternion)
    light_object_2.rotation_mode = "QUATERNION"
    light_object_2.rotation_quaternion = direction_2.to_track_quat('-Z', 'Y')
    print(light_object_2.rotation_quaternion)
    # light_object_2.data.cycles.cast_shadow = False
    bpy.context.object.data.cycles.cast_shadow = False

    # update scene, if needed
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()

def get_calibration_matrix_K_from_blender(camd):
    '''
    https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    Function to obtain camera intrinsics
    '''
    # camd.sensor_fit = "VERTICAL"
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # print('vertical')
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm


    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

def get_camera_intrinsics(cam_obj):
    '''
    https://www.rojtberg.net/1601/from-blender-to-opencv-camera-and-back/
    '''
    # get the relevant data
    cam = cam_obj.data
    scene = bpy.context.scene
    # assume image is not scaled
    assert scene.render.resolution_percentage == 100
    # assume angles describe the horizontal field of view
    assert cam.sensor_fit != 'VERTICAL'

    f_in_mm = cam.lens
    sensor_width_in_mm = cam.sensor_width
    # print(f_in_mm, sensor_width_in_mm)

    w = scene.render.resolution_x
    h = scene.render.resolution_y

    pixel_aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x

    f_x = f_in_mm / sensor_width_in_mm * w
    f_y = f_x * pixel_aspect

    # yes, shift_x is inverted.
    c_x = w * (0.5 - cam.shift_x)
    # and shift_y is still a percentage of width..
    c_y = h * 0.5 + w * cam.shift_y

    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0,   0,   1]])

    return K

def set_camera_extrinsics(cam_obj, car_obj, location):

    '''
    Sets camera location and look at car_obj
    '''

    cam_obj.location = location

    camera_look_at_object(cam_obj, car_obj)


def set_rendering_settings():
    '''
    Sets rendering settings for background
    '''
    bpy.context.scene.render.film_transparent = True
    # bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.resolution_percentage = 100
    cam_obj = bpy.data.objects['Camera']

    bpy.context.scene.camera = cam_obj
    bpy.context.scene.cycles.samples = 250
    bpy.context.scene.frame_end = 1
    # bpy.context.scene.use_denoising = True
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0.00

    bpy.context.scene.view_settings.look = 'Medium Contrast'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.cycles.max_bounces = 4
    bpy.context.scene.cycles.caustics_reflective = False
    bpy.context.scene.cycles.caustics_refractive = False
    bpy.context.scene.cycles.sample_clamp_indirect = 0
    bpy.context.scene.view_layers["View Layer"].use_pass_object_index = True
    bpy.context.scene.view_layers["View Layer"].use_pass_z = True
    bpy.context.scene.cycles.device = 'GPU'


def gen_helical_path(num_images):

    '''
    Function to generate helical path around object
    '''

    highest_z = 1 + 3.0*random.random()
    base_radius = 1.0 + 0.5 * np.random.rand()

    u = np.linspace( -highest_z*np.pi, highest_z*np.pi, num_images)
    radius = base_radius + 5* np.abs(np.sin(u) - 0.03) #+ 0.5*np.random.random(num_images)

    x = radius * (np.cos(u))
    y = radius * (np.sin(u))
    z = u / np.pi

    return x, y, z
    # plot(x,y,z,'r');

def fix_shapenet_lighting(category_object):
    '''
    Fix lighting and material issues
    '''

    category_object.data.use_auto_smooth = False
    category_object.modifiers.data.modifiers.new(name = "edgesplit",type="EDGE_SPLIT")
    category_object.pass_index = 1.0

    for material in list(category_object.data.materials):

        # Roughness
        material.node_tree.nodes["Principled BSDF"].inputs[7].default_value = 0.0
        # Specular
        material.node_tree.nodes["Principled BSDF"].inputs[5].default_value = 5
        # Metallic
        material.node_tree.nodes["Principled BSDF"].inputs[4].default_value = 0.7



def convert_extrinsics_2_unity(camera_json):
    '''
    Convert camera extrinsics from blender to unity
    https://gist.github.com/piranha771/e97c773fc050bc6387d36a080c4dd132
    Not tested properly.......
    '''

    camera_json_unity = camera_json.copy()

    cam_position = camera_json['position']
    cam_rotation = camera_json['rotation']

    camera_json_unity['position']['x']  = -cam_position['x']
    camera_json_unity['position']['y']  = cam_position['z']
    camera_json_unity['position']['z']  = cam_position['y']

    camera_json_unity['rotation']['z']  = -cam_rotation['y']
    camera_json_unity['rotation']['y']  = cam_rotation['z']

    return camera_json_unity

def project_by_object_utils(cam, point):
    '''
    https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    Projects 3D point to image plane and returns its location
    '''

    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))


def get_3x4_RT_matrix_from_blender(cam):
    '''
    https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera

    '''
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_3x4_P_matrix_from_blender(cam):

    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K @ RT, K, RT


def save_keypoint_locations(kp_json, index, depth, mask):
    '''
    Function to save keypoints
    '''

    save_path = os.path.join(output_file_path, "") + kp_json["class_id"] + "/" + kp_json["model_id"] + "/" + "frame_%08d_KeyPoints.npy" % index

    P, K, RT = get_3x4_P_matrix_from_blender(bpy.data.objects['Camera'])
    # Apply rotation to align keypoints and model
    rotation_fix = mathutils.Matrix.Rotation(np.radians(90.0), 4, 'X')
    cam_obj = bpy.data.objects["Camera"]
    keypoint_matrix = []
    j = 0

    avg_depth = np.mean(depth[mask > 0.5])
    print(avg_depth)

    kp_array = np.zeros((len(keypoint_dict.keys()), 3))
    for kp in kp_json['keypoints']:

        point = mathutils.Vector((kp['xyz'][0], kp['xyz'][1], kp['xyz'][2], 1))
        # print("before rot", point)
        point = rotation_fix @ point
        # print("After rotation", point)
        point_3d_camera_frame =  RT @ point

        # print(point_3d_camera_frame)

        point_2d = project_by_object_utils(cam_obj, point)

        if (point_2d[1] < depth.shape[0]) and (point_2d[1] >= 0) and (point_2d[0] >=0) and (point_2d[0] < depth.shape[1]):

            point_vis_depth = depth[int(point_2d[1]), int(point_2d[0])]
            depth_difference = np.abs(point_vis_depth - point_3d_camera_frame[2])
            mask_value = (mask[int(point_2d[1]), int(point_2d[0])] >= 0.5) * 1
            print(depth_difference, mask_value)
        else:
            print("pt outside")
            depth_difference = 100


        if depth_difference < 0.1:
            point_2d = np.array([point_2d[0], point_2d[1], 1])
        else:
            point_2d = np.array([point_2d[0], point_2d[1], 0])
        # else:
        #     point_2d = np.array([point_2d[0], point_2d[1], 1])

        keypoint_matrix.append(point_2d)

        # Check to mask sure semantic id is same
        # assert kp["semantic_id"] == j, "Semantic id error"
        # j = j + 1
        
        if kp["semantic_id"] in keypoint_dict.keys():
            kp_array[keypoint_dict[kp["semantic_id"]], 0] = point_2d[0]
            kp_array[keypoint_dict[kp["semantic_id"]], 1] = point_2d[1]
            kp_array[keypoint_dict[kp["semantic_id"]], 2] = point_2d[2]

    # keypoint_matrix = np.stack(keypoint_matrix, axis = 0)
    keypoint_matrix = kp_array
    
    print(keypoint_matrix)

    np.save(save_path, keypoint_matrix)

def set_sequence_lighting():
    '''
    Function to randomize sequence lighting
    '''
    light_1 = bpy.data.objects["light_1"]
    light_2 = bpy.data.objects["light_2"]

    x, y = 5 * np.random.rand(2) - 2.5

    z_square = 10 - (x**2 + y**2)
    z = np.sqrt(z_square)
    
    # if np.random.rand() < 0.5:
        
    
    light_1.location = (x, y, z)
    light_1.rotation_mode = "QUATERNION"
    direction = mathutils.Vector((0.0,0.0, 0.0)) - light_1.location
    light_1.rotation_quaternion = direction.to_track_quat('-Z', 'Y')
    light_1.data.energy = 40000 - 3000 * np.random.rand()
    light_1.data.shadow_soft_size = 3

    light_2.location = (-x, -y, -z)
    light_2.rotation_mode = "QUATERNION"
    direction = mathutils.Vector((0.0,0.0, 0.0)) - light_2.location
    light_2.rotation_quaternion = direction.to_track_quat('-Z', 'Y')
    light_2.data.energy = 100
    light_2.data.shadow_soft_size = 0

    bpy.context.object.data.cycles.max_bounces = 256

def gen_data(path, kp_json):
    '''
    Function to generate train data
    '''

    # print(kp_json)
    # clear meshes that are present
    clear_mesh()
    # Get camera helical path
    randomlist = [random.randint(0, number_backgrounds - 1) for i in range(num_images)]

    x, y, z =  gen_helical_path(num_images)


    bpy.ops.import_scene.obj(filepath=path)

    print(list(bpy.data.objects))
    # bpy.ops.import_scene.obj(filepath=g_shapenet_path + '02958343/27e267f0570f121869a949ac99a843c4/model.obj')

    # Extracting objects
    car_obj = bpy.data.objects['model_normalized']

    print('Rotation')
    cam_obj = bpy.data.objects['Camera']

    fix_shapenet_lighting(car_obj)
    # node_setting_init()
    save_directory = output_file_path + kp_json["class_id"] + "/" + kp_json["model_id"] + "/"
    set_image_path(save_directory)

    K = get_calibration_matrix_K_from_blender(cam_obj.data)
    print('[INFO] Camera intrinsics: \n')
    print(K)
    set_sequence_lighting()


    for i in range(len(x)):

        bpy.context.scene.frame_set(i)

        set_camera_extrinsics(cam_obj, car_obj, mathutils.Vector((x[i], y[i], z[i])))

        # set_camera_extrinsics(cam_obj, car_obj, mathutils.Vector((0.95, -1.3, -0.163)))
        # set_camera_extrinsics(cam_obj, car_obj, mathutils.Vector((1.3, -0.6, 1.2)))

        bpy.ops.object.select_all(action='TOGGLE')

        camera_quaternion = cam_obj.rotation_quaternion
        camera_position = cam_obj.location


        print('##################################')
        print('[INFO] Camera extrinsics')
        # print(camera_quaternion, camera_position)
        camera_json = {"position":{"x": camera_position.x ,"y":camera_position.y, "z":camera_position.z},"rotation":{"x":camera_quaternion.x,"y":camera_quaternion.y,"z":camera_quaternion.z,"w":camera_quaternion.w}}
        print(camera_json)
        print('##################################')


        image_node = bpy.context.scene.node_tree.nodes[0]
        image_node.image = bpy.data.images.load(backgrounds[randomlist[i]])
        bpy.ops.render.render(write_still=False)
        # break

        # Save json
        json_file_name = "frame_%08d_CameraPose.json" % i
        json_file_name = os.path.join(save_directory, json_file_name)
        print(json_file_name)
        with open(json_file_name, "w") as fp:
            json.dump(camera_json, fp)
        # Read depth image after it is saved
        depth_image_path = save_directory + 'frame_%08d_Depth_00.exr' % i
        mask_image_path = save_directory + "frame_%08d_Mask_00.png" % i
        depth_image = imageio.imread(depth_image_path)[:, :, 0]

        # Mask is 16 bit
        mask_image = imageio.imread(mask_image_path) / 65535.0
        # Save keypoints
        save_keypoint_locations(kp_json, i, depth_image, mask_image)


def check_keypoint_function(kps, num_kps, gt_and_equal = False):

    if gt_and_equal:
        if len(kps) >= num_kps:
            print("great")
            return True
    else:
        if len(kps) == num_kps:
            return True

    return False

if __name__ == "__main__":

    '''
    Tasks

    Set cam intrinsics - Done
    Check scale - Done
    Helical path - Done
    Save camera positions - Done
    Color correction / lighting - Done
    Save mask and depth - Done
    Change backgrounds repeatedly - Done

    Check with warp function - Done
    Add key points -
    Clean code -

    Change blender coordinate to unity x
    Change code to run for multiple models -
    '''

    num_images = g_number_images
    background_image_folder = g_background_image_path

    global output_file_path, number_backgrounds, backgrounds
    output_file_path = g_output_path
    # np.random.seed

    # path join appends / if not present
    background_images_location = os.path.join(background_image_folder,"") +  './**'
    output_file_path = os.path.join(output_file_path, "")
    backgrounds = glob.glob(background_images_location)

    number_backgrounds = len(backgrounds)
    assert number_backgrounds > 0, "0 backgrounds available in the provided folder location"

    car_kp_json_file = g_kp_json
    with open(car_kp_json_file) as fp:
        car_kp_json = json.load(fp)
    model = 1


    global keypoint_dict, gt_and_equal, num_kps
    
    gt_and_equal = g_gt_and_equal
    keypoint_dict = g_keypoint_dict
    num_kps = g_num_kps

    init_all()
    # print(car_kp_json[model]['model_id'])

    for i in range(len(car_kp_json)):
        path_obj = g_shapenet_path + "/" + car_kp_json[i]['class_id'] + "/" + car_kp_json[i]['model_id'] + "/models/model_normalized.obj"

        print(i)
        # Check if model exists
        # Generate data for the model if it exists
        if os.path.exists(path_obj):
            print(len(car_kp_json[i]["keypoints"]), num_kps, gt_and_equal)
            if check_keypoint_function(car_kp_json[i]["keypoints"], num_kps, gt_and_equal):
                # if i > 3:
                gen_data(path_obj, car_kp_json[i])
                    
                    # break
            # if len(car_kp_json[i]["keypoints"]) == 22:
            #     gen_data(path_obj, car_kp_json[i])

    # bpy.ops.wm.quit_blender()