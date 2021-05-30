import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import glob
import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import json
import argparse
from scipy.stats import multivariate_normal
from skimage.transform import resize
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import PIL
import kornia

class Keypoint_dataset(Dataset):
    '''
    Keypoint dataset for training
    '''

    def __init__(self, dataset_path, train, resize_dimension = 64, transform = None):

        self.train = train
        self.resize_dimension = resize_dimension
        self.transform = transform
        self.name_dictionary = self.load_names(dataset_path, train)
        print("Number of images: ", len(self.name_dictionary["images"]))
        print('Data retrieved')


    def __len__(self):

        return len(self.name_dictionary["images"])

    def __getitem__(self, index):

        prob_rotate = random.random()
        prob_crop = random.random()

        keypoint_view = np.load(self.name_dictionary["keypoints"][index])
        image_view = (cv2.cvtColor(cv2.imread(self.name_dictionary["images"][index]),cv2.COLOR_BGR2RGB) * 255).astype("uint8")
        h, w, c = image_view.shape
        image_view = resize(image_view, (self.resize_dimension, self.resize_dimension), preserve_range=True)[:, :, :3].astype("uint8")

        # print(image_view.astype("uint8"))
        keypoint_view[:, 0] = keypoint_view[:, 0] * self.resize_dimension / w
        keypoint_view[:, 1] = keypoint_view[:, 1] * self.resize_dimension / h

        alpha = 180#random.randint(-35, 35)
        angle_rot = alpha * np.pi / 180
        rot_mat = np.array([
            [np.cos(angle_rot), -np.sin(angle_rot)],
            [np.sin(angle_rot), np.cos(angle_rot)]
        ])

        if prob_rotate > 0.5:
            keypoint_view[:, 0] = keypoint_view[:, 0] - 32
            keypoint_view[:, 1] = keypoint_view[:, 1] - 32
            keypoint_view[:,:2] = keypoint_view[:,:2] @ rot_mat
            keypoint_view[:, 0] = keypoint_view[:, 0] + 32
            keypoint_view[:, 1] = keypoint_view[:, 1] + 32


        data = {}
        data["views"] = image_view
        data["heatmaps"] = self.generate_target_heatmaps_batch(keypoint_view, (self.resize_dimension, self.resize_dimension))
        data["keypoints"] = keypoint_view

        transform_tensor = self.ToTensor()
        data = transform_tensor(data, self.resize_dimension, train = self.train)

        ## Adding the random transformations here
        if prob_rotate > 0.5:
            data["views"] = T.ToPILImage(mode="RGB")(data["views"])
            data["views"] = TF.rotate(data["views"], alpha)
            data["views"] = T.ToTensor()(data["views"])

        if prob_crop > 0.8:
            window = random.randint(16,32)
            top = random.randint(0, 64-window)
            left = random.randint(0, 64-window)
            data["views"][:,top:top+window,left:left+window] = 0.0
            data["heatmaps"][:,top:top+window,left:left+window] = data["heatmaps"][:,top:top+window,left:left+window]*.0

        return data

    def generate_heatmap(self, kp, image_size):
        '''
        Generate a single heatmap
        '''

        pos = np.dstack(np.mgrid[0:image_size[0]:1, 0:image_size[1]:1])
        # kp => x, y, visibility
        rv = multivariate_normal(mean=[kp[1], kp[0]], cov = [1,1])
        heatmap = rv.pdf(pos)

        heatmap = heatmap / heatmap.max()

        if kp[2] == 0:
            heatmap *= 0

        return heatmap

    def generate_target_heatmaps_batch(self, kps, image_size):
        '''
        Generate target heatmaps from keypoints
        '''
        heatmaps = []

        for i in range(kps.shape[0]):
            heatmaps.append(self.generate_heatmap(kps[i, :], image_size))

        target_heatmap = np.stack(heatmaps, axis = 0)

        return target_heatmap

    def load_names(self, path, train = 1):
        '''
        Load names of image, mask, and camera pose file names
        '''
        if train == 1:
            data_file_name = 'train.txt'
        elif train == 2:
            data_file_name = 'val.txt'
        else:
            data_file_name = 'test.txt'

        file_path = os.path.join(path, data_file_name)

        fp = open(file_path, 'r')

        images = []
        keypoints = []

        for line in fp:

            fields = line.split('/')
            directory_name = fields[0]
            image_name = fields[1]

            image_number = int(image_name.split("_")[1])
            keypoint_name = "frame_%08d_KeyPoints.npy" % image_number

            images.append(os.path.join(path, directory_name, image_name).split("\n")[0])
            keypoints.append(os.path.join(path, directory_name, keypoint_name))

        return {"images": images, "keypoints": keypoints}

    class ToTensor(object):

        def __call__(self, data_dictionary, resize_dimension = 64, train = 1):

            if train == 1:
                data_augmentation_transform = transforms.Compose([
                                            transforms.ToPILImage(),
                                            #transforms.ColorJitter(brightness=0.0, contrast=0.05, saturation=0.05, hue = 0.3),
                                            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.05, hue = 0.3),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                data_augmentation_transform = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            # data_sample["heatmaps"] = ((data_dictionary["heatmaps"]))
            # data_sample["views"] = tensor_transform(resize_transform(pil_transform(data_dictionary["views"])))

            data_sample = {}
            #print(data_dictionary["views"].shape)

            data_sample["views"] = (data_augmentation_transform(data_dictionary["views"]))
            data_sample["heatmaps"] = torch.from_numpy(data_dictionary["heatmaps"])
            # data_sample["views"] = torch.from_numpy(data_dictionary["views"].transpose(2, 0, 1)) / 255.0
            data_sample["keypoints"] = torch.from_numpy(data_dictionary["keypoints"])
            return data_sample


def keypoint_visualizer(im, kps):

    plt.imshow(im)

    for i in range(kps.shape[0]):

        # If visibility is 1 display point in red
        if kps[i, 2] == 1:
            plt.plot(kps[i, 0], kps[i, 1], 'ro')
        else:
            plt.plot(kps[i, 0], kps[i, 1], 'bo')

    plt.show()

if __name__ == "__main__":

    ################################## Argument parser ###########################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset path", required=True)
    args = parser.parse_args()

    ##############################################################################################################

    dataset_path = args.dataset

    data_set = Keypoint_dataset(dataset_path, train = 1)
    data_loader = DataLoader(data_set, batch_size = 2, shuffle=True)

    data = data_set[20]

    # print(data["views"].min())
    # print(data["keypoints"].numpy())
    plt.subplot(121)
    plt.imshow(data["heatmaps"][2])
    plt.subplot(122)
    plt.imshow(data["views"].permute(1, 2, 0))
    plt.show()
    # keypoint_visualizer(data["views"].permute(1, 2, 0), data["keypoints"].numpy())

    # for batch, data_sample in enumerate(data_loader):
        # print(batch, data_sample["views"].min())
