import numpy as np
import matplotlib.pyplot as plt
import imageio
import argparse
import os
import glob
import random
import tqdm

def extract_bounding_box(mask, threshold = 10):

    '''
    Function to extract bounding box for an image given segmentation mask.

    '''

    random_threshold = threshold * np.random.rand()
    y_min = (np.min(np.where(mask >= 0.5)[0]))
    x_min = (np.min(np.where(mask >= 0.5)[1]))

    y_max = (np.max(np.where(mask >= 0.5)[0]))
    x_max = (np.max(np.where(mask >= 0.5)[1]))

    y_image, x_image = mask.shape

    if (x_min - random_threshold > 0):
        x_min -= random_threshold

    if (x_max + random_threshold < x_image):
        x_max += random_threshold

    if (y_min - random_threshold > 0):
        y_min -= random_threshold

    if (y_max + random_threshold < y_image):
        y_max += random_threshold


    return int(y_min), int(x_min), int(y_max), int(x_max)

def write_list_2_file(list_set, filename):

    with open(filename,"w") as fp:
        for list_item in list_set:
            fp.write(list_item + "\n")

    fp.close()

def split_into_sets(list_images, output_directory):
    '''
    Split images into train/val/test split and save the txt file
    '''

    random.shuffle(list_images)
    new_list = []
    for file_image in list_images:
        new_list.append(file_image.split("/")[-2]+ "/" + file_image.split("/")[-1])

    images_train = new_list[:int(len(new_list)*0.8)]
    images_val = new_list[int(len(new_list)*0.8):int(len(new_list)*0.9)]
    images_test = new_list[int(len(new_list)*0.9):]

    images_train.sort()
    images_val.sort()
    images_test.sort()

    write_list_2_file(images_train, output_directory + "train.txt")
    write_list_2_file(images_val, output_directory + "val.txt")
    write_list_2_file(images_test, output_directory + "test.txt")


if __name__ == "__main__":

    ################ Argument parser ########################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input directory path", required=True)
    parser.add_argument("--output", help="Output directory path", required=True)
    parser.add_argument("--threshold", help="Threshold for cut", required=False, default=10)

    args = parser.parse_args()

    ##########################################################################################


    input_directory = os.path.join(args.input,"")
    output_directory = os.path.join(args.output, "")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    list_images = glob.glob(input_directory + "**/*Color_00*.png")

    split_into_sets(list_images, output_directory)

    list_images.sort()


    pbar = tqdm.tqdm(total = len(list_images))

    for image_name in list_images:

        image_directory = image_name.split("/")[-2]

        if not os.path.exists(output_directory + image_directory):
            os.makedirs(output_directory + image_directory)

        image_number = int(image_name.split("/")[-1].split("_")[1])
        #print(image_directory, image_number)

        kp_name = "frame_%08d_KeyPoints.npy" % image_number
        image_name = "frame_%08d_Color_00.png" % image_number
        mask_name = "frame_%08d_Mask_00.png" % image_number

        kps = np.load(input_directory + "/" + image_directory + "/" + kp_name)
        image = plt.imread(input_directory + "/" + image_directory + "/" + image_name)
        mask = plt.imread(input_directory + "/" + image_directory + "/" + mask_name)


        y_min, x_min, y_max, x_max = extract_bounding_box(mask, args.threshold)

        kps[:, 0] -= x_min
        kps[:, 1] -= y_min

        cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max), :]

        plt.imsave(output_directory + image_directory + "/" + image_name, cropped_image)
        np.save(output_directory + image_directory + "/" + kp_name, kps)
        pbar.update(1)

    pbar.close()
