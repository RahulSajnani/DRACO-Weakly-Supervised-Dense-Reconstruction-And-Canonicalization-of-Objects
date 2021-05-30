import numpy as np
import matplotlib.pyplot as plt
import cv2
from visualize_nocs import *

if __name__=="__main__":

    nocs_image = plt.imread("../../../../test/82c224e707f63304acb3c21764457201/frame_00000004_NOXRayTL_00.png")[:, :, :3]
    #print(nocs_image)

    rotation_mat = np.array([[ 0.10401288, -0.97621297, -0.13562396],
                            [-0.98353821, -0.09394213, -0.07730554],
                            [ 0.06216511,  0.14517352, -0.98294882]])
    
    
    for i in range(3):
        norm = np.sqrt(np.square(rotation_mat[:, i]).sum())
        rotation_mat[:, i] = rotation_mat[:, i] / norm

    nocs_image = rotate_nocs(nocs_image, rotation_mat)
    plt.imshow(nocs_image)
    plt.show()

    plt.imsave("./nocs_rotated.png", nocs_image)
