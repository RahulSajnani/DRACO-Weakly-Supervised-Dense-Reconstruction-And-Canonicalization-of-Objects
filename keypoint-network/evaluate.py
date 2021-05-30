import numpy as np
import torch, os
import argparse
import glob
import matplotlib.pyplot as plt
#from kornia import *
from models import hourglass
from skimage.transform import resize

def preprocess_image(image, dim = 64):
    '''
    Function to preprocess image
    '''
    h, w, c = image.shape
    image_out = resize(image, (dim, dim))[:, :, :3]

    return image_out, {"scale_y": h / dim, "scale_x": w / dim}

def heatmap_2_keypoints(heatmaps, scale_dict):
    '''
    Function to obtain keypoint locations from heatmaps
    '''
    if heatmaps.is_cuda:
        heatmaps_numpy = heatmaps.detach().cpu().numpy()[0]
    else:
        heatmaps_numpy = heatmaps.detach().numpy()[0]

    keypoint_list = []

    for i in range(heatmaps_numpy.shape[0]):

        heatmap = heatmaps_numpy[i]
        ind = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
        #print(ind)
        #print(heatmap[ind])
        vis = heatmap[ind]
        keypoint_list.append(np.array([ind[1], ind[0], vis]))


    #keypoint_locations = spatial_soft_argmax2d(heatmaps, normalized_coordinates = False)
    keypoint_locations = np.stack(keypoint_list, 0).astype("float")
    keypoint_locations[:, 0] *= scale_dict["scale_x"]
    keypoint_locations[:, 1] *= scale_dict["scale_y"]

    return keypoint_locations

if __name__ == "__main__":

    ################################# Argument Parser #####################################

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help="Path to image folder", required=True)
    parser.add_argument("--model", help="Path to pretrained network", required=True)
    parser.add_argument("--output", help="Path to output folder", required=True)

    args = parser.parse_args()

    #######################################################################################

    print(args.input)
    images_list = glob.glob(os.path.join(args.input, "**.jpg"))
    images_list.sort()

    print(images_list)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    net = hourglass.hg(num_classes = 22, num_stacks = 3, num_blocks = 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(args.model))
        net.cuda()
    else:
        net.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))

    net.eval()
    #print(images_list)
    for image_name in images_list:

        #print(image_name)
        #dir_image = os.path.join("./", *image_name.split("/")[:-1])
        #print((image_name.split(".")[-2].split("_")[-3]))
        #image_number = int(image_name.split(".")[-2].split("_")[-3])

        image_view = plt.imread(image_name)
        image_view, scale_dict = preprocess_image(image_view)
        #print(image_view.max())
        #kp_name = os.path.join(dir_image, "frame_%08d_KeyPoints.npy" % image_number)
        #kp = np.load(kp_name)
        #print(kp)

        if torch.cuda.is_available():
            image_tensor = torch.from_numpy(image_view.transpose(2, 0, 1)).unsqueeze(0).cuda()
        else:
            image_tensor = torch.from_numpy(image_view.transpose(2, 0, 1)).unsqueeze(0)

        output_heatmaps = net(image_tensor.float())

        #print(len(output_heatmaps))
        keypoint_array = heatmap_2_keypoints(output_heatmaps[-1], scale_dict)

        #print(keypoint_array)
        kp_output_name = os.path.join(args.output, image_name.split("/")[-1].split(".")[-2] + ".npy")
        print(kp_output_name)
        np.save(kp_output_name, keypoint_array)



