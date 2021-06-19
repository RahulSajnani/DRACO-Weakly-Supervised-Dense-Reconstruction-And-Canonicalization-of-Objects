import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import torch


def sigmoid_2_depth(normalised_depth, scale_factor=10):
    '''
    Converts network sigmoid prediction to depth
    '''
    depth = scale_factor * (1 - normalised_depth)

    return depth


def pose_to_transformation_matrix(pose):

    '''
    Converts XNOCS pose to transformation matrix
    '''

    a = pose['rotation']['x']
    b = pose['rotation']['y']
    c = pose['rotation']['z']

def construct_camera_matrix(focal, cx, cy):
    '''
    Construct camera intrinsic matrix
    '''
    K = np.array([[focal,     0,    cx],
                  [    0, focal,    cy],
                  [    0,     0,    1]])

    return K


def read_json_file(path):
    '''
    Read json file
    '''
    json_data = []

    with open(path) as fp:
        for json_object in fp:
            json_data.append(json.loads(json_object))

    return json_data

def plot_split_image_dictionary(images_dictionary):

    '''
    plot function for split images
    '''

    pass

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


if __name__ == "__main__":

    image = cv2.imread('frame_00000000_Color_00.png')
    camera_pose = read_json_file('frame_00000000_CameraPose.json')
    # images_dictionary = split_image(image, camera_pose, 5)



    # plt.imshow(images_dictionary['view_-2'])
    # plt.show()
    pass
