from __future__ import division
import argparse
import numpy as np
from glob import glob
# from joblib import Parallel, delayed
from pebble import ProcessPool
from tqdm import tqdm
import os
import cv2
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",   type=str, required=True, help="Path to where the dataset it stored")
parser.add_argument("--dataset_name",  type=str, required=True, choices=["cars", "cars_blender", "epfl", "cars_blender_canonical", "cars_blender_nocs"])
parser.add_argument("--dump_root",     type=str, required=True, help="Path to store the processed data")
parser.add_argument("--seq_length",    type=int, required=True, help="length of each training sequence")

# parser.add_argument("--seq_length",    type=int, required=True, help="length of each training sequence")
parser.add_argument("--depth",    type=int, default = 1, required=False, help="Depth")
parser.add_argument("--canonical",    type=int, default = 0, required=False, help="Canonicalize")

parser.add_argument("--img_height",    type=int, default=480,   help="image height")
parser.add_argument("--img_width",     type=int, default=640,   help="image width")
parser.add_argument("--num_threads",   type=int, default=4,     help="number of threads to use")
args = parser.parse_args()

def concat_image_seq(seq):
    '''
        Function to horizontanly stack the sequence images
        Input
        seq: Input sequence of images
        return 
        result: Horizontally concatenated sequence
    '''
    for i, img in enumerate(seq):
        if i == 0:
            result = img
        else:
            result = np.hstack((result, img))
    return result


def concat_keypoints_seq(seq):
    '''
        Function to concatenate the sequence of keypoints parameters
        Input
        seq: Input sequence of keypoints
        return
        result: Concatenated keypoints matrix
    '''
    keypoints_seq = []
    for keypoints in seq:
        keypoints_seq.append(keypoints)
    
    keypoints_seq = np.stack(keypoints_seq)
    # print(keypoints_seq.shape)
    return keypoints_seq


def concat_extrinsic_seq(seq):
    '''
        Function to concatenate the sequence of extrinsic parameters
        Input
        seq: Input sequence of extrinsics in JSON format
        return
        result: Concatenated JSON extrinsics sequence
    '''
    extrinsics_seq = ""
    for extrinsic in seq:
        extrinsics_seq += extrinsic
        extrinsics_seq += "\n"
    return extrinsics_seq


def dump_sample_epfl(sample_index, dump_root):
    '''
        Function to write the processed sample
        Input
        sample_index: Index of the target frame that needs to be sampled
        dump_root   : Path to the root of the processed outputs
    '''
    # if sample_index % 2000 == 0:
    #     print('Progress %d/%d....' % (sample_index, data_loader.num_frames))
   
    example = data_loader.get_sample_with_idx(sample_index)
    if example == False:
        return


    dump_dir = os.path.join(dump_root, example['folder_name'])

    try: 
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise

    color00_image_seq = concat_image_seq(example['color00_img_seq'])
    dump_img_file   = f"{dump_dir}/view_{example['file_name']}.jpg"
    cv2.imwrite(dump_img_file, color00_image_seq.astype(np.uint8))


def dump_sample(sample_index, dump_root, blender = 0):
    '''
        Function to write the processed sample
        Input
        sample_index: Index of the target frame that needs to be sampled
        dump_root   : Path to the root of the processed outputs
    '''
    # if sample_index % 2000 == 0:
    #     print('Progress %d/%d....' % (sample_index, data_loader.num_frames))
   
    example = data_loader.get_sample_with_idx(sample_index)
    if example == False:
        return


    dump_dir = os.path.join(dump_root, example['folder_name'])

    try: 
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise

    color00_image_seq = concat_image_seq(example['color00_img_seq'])
    depth_image_seq = concat_image_seq(example['depth_img_seq'])
    extrinsics_seq = concat_extrinsic_seq(example['extrinsics_seq'])

    dump_img_file   = f"{dump_dir}/view_{example['file_name']}.jpg"
    dump_depth_file = f"{dump_dir}/depth_{example['file_name']}.jpg"
    dump_cam_file   = f"{dump_dir}/CameraPose_{example['file_name']}.json"
    
    if use_blender:
        mask00_image_seq = concat_image_seq(example['mask00_img_seq'])
        dump_mask_file = f"{dump_dir}/mask_{example['file_name']}.jpg"
        cv2.imwrite(dump_mask_file, mask00_image_seq.astype(np.uint8))
        
        keypoints_seq = concat_keypoints_seq(example['keypoints_seq'])
        dump_kps_file = f"{dump_dir}/keypoints_{example['file_name']}.npy"
        np.save(dump_kps_file, keypoints_seq)
        
    else:
        color01_image_seq = concat_image_seq(example['color01_img_seq'])
        nox_image_seq = concat_image_seq(example['nox_img_seq'])
        nox_image_seq[np.where(nox_image_seq==255)] = 0
        nox_image_seq[np.where(nox_image_seq!=0)] = 255
        opp_extrinsics_seq = concat_extrinsic_seq(example['opp_extrinsics_seq'])

        dump_nox_file   = f"{dump_dir}/mask_{example['file_name']}.jpg"
        cv2.imwrite(dump_nox_file, nox_image_seq.astype(np.uint8))
    
    # Dumping the forward image

    cv2.imwrite(dump_img_file, color00_image_seq.astype(np.uint8))
    cv2.imwrite(dump_depth_file, depth_image_seq.astype(np.uint8))
    
    with open(dump_cam_file, 'w') as f:
        f.write(extrinsics_seq)

def dump_sample_canonical(sample_index, dump_root):
    '''
        Function to write the processed sample
        Input
        sample_index: Index of the target frame that needs to be sampled
        dump_root   : Path to the root of the processed outputs
    '''
    # if sample_index % 2000 == 0:
    #     print('Progress %d/%d....' % (sample_index, data_loader.num_frames))
   
    example = data_loader.get_sample_with_idx(sample_index)
    if example == False:
        return
    dump_dir = os.path.join(dump_root, example['folder_name'])

    try: 
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise

    color00_image_seq = concat_image_seq(example['color00_img_seq'])
    nox00_image_seq = concat_image_seq(example['nox00_img_seq'])
    depth_image_seq = concat_image_seq(example['depth_img_seq'])
    extrinsics_seq = concat_extrinsic_seq(example['extrinsics_seq'])

    dump_img_file   = f"{dump_dir}/view_{example['file_name']}.jpg"
    dump_nox_file   = f"{dump_dir}/nocs_{example['file_name']}.jpg"
    dump_cam_file   = f"{dump_dir}/CameraPose_{example['file_name']}.json"
    
    if depth_image_seq is not None:
        dump_depth_file = f"{dump_dir}/depth_{example['file_name']}.tiff"
    
    if use_blender:
        mask00_image_seq = concat_image_seq(example['mask00_img_seq'])
        dump_mask_file = f"{dump_dir}/mask_{example['file_name']}.jpg"
        cv2.imwrite(dump_mask_file, mask00_image_seq.astype(np.uint8))
        
        if canonical:
            keypoints_seq = concat_keypoints_seq(example['keypoints_seq'])
            dump_kps_file = f"{dump_dir}/keypoints_{example['file_name']}.npy"
            np.save(dump_kps_file, keypoints_seq)
            
            c3dpo_cam_coords_seq = concat_keypoints_seq(example['c3dpo_cam_coords_seq'])
            c3dpo_cam_coords_seq_file = f"{dump_dir}/c3dpo_{example['file_name']}_cam_coords.npy"
            np.save(c3dpo_cam_coords_seq_file, c3dpo_cam_coords_seq)
            
            c3dpo_rotation_seq = concat_keypoints_seq(example['c3dpo_rotation_seq'])
            c3dpo_rotation_seq_file = f"{dump_dir}/c3dpo_{example['file_name']}_rotation.npy"
            np.save(c3dpo_rotation_seq_file, c3dpo_rotation_seq)
            
    
    # Dumping the forward image

    cv2.imwrite(dump_img_file, color00_image_seq.astype(np.uint8))
    cv2.imwrite(dump_nox_file, nox00_image_seq.astype(np.uint8))
    
    if depth:
        imageio.imwrite(dump_depth_file, depth_image_seq)
        # cv2.imwrite(dump_depth_file, depth_image_seq.astype(np.uint8))
    
    with open(dump_cam_file, 'w') as f:
        f.write(extrinsics_seq)


def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    
    global data_loader, use_blender, canonical, depth
    if args.dataset_name == 'cars':
        from cars.cars_loader import CarsLoader
        data_loader = CarsLoader(dataset_dir=args.dataset_dir,
                                 seq_length=args.seq_length)
        use_blender = 0
    elif args.dataset_name == 'cars_blender':
        from cars_blender.cars_loader import CarsLoader
        data_loader = CarsLoader(dataset_dir=args.dataset_dir,
                                 seq_length=args.seq_length)
        use_blender = 1
    
    elif args.dataset_name == 'cars_blender_canonical':
        from cars_blender_canonical.cars_loader import CarsLoader
        data_loader = CarsLoader(dataset_dir=args.dataset_dir,
                                 seq_length=args.seq_length,
                                 depth=args.depth,
                                 canonical=args.canonical)
        use_blender = 1
        canonical = args.canonical
        depth = args.depth

    elif args.dataset_name == 'cars_blender_nocs':
        from cars_blender_nocs.cars_loader import CarsLoader
        data_loader = CarsLoader(dataset_dir=args.dataset_dir,
                                 seq_length=args.seq_length,
                                 depth=args.depth,
                                 canonical=args.canonical)
        use_blender = 1
        canonical = args.canonical
        depth = args.depth

    
    elif args.dataset_name == 'epfl':
        from epfl.epfl_loader import EPFLLoader
        data_loader = EPFLLoader(dataset_dir=args.dataset_dir,
                                 seq_length=args.seq_length)


    with ProcessPool(max_workers=args.num_threads) as pool:
        if args.dataset_name == 'epfl':
            tasks = pool.map(dump_sample_epfl, range(data_loader.num_frames), [args.dump_root]*data_loader.num_frames)
        
        elif args.dataset_name == "cars_blender_canonical" or args.dataset_name == "cars_blender_nocs":
            tasks = pool.map(dump_sample_canonical, range(data_loader.num_frames), [args.dump_root]*data_loader.num_frames)
        
        else:
            tasks = pool.map(dump_sample, range(data_loader.num_frames), [args.dump_root]*data_loader.num_frames, [use_blender]*data_loader.num_frames)

        try:
            for _ in tqdm(tasks.result(), total=data_loader.num_frames):
                pass
        except KeyboardInterrupt as e:
            tasks.cancel()
            raise e

    # Split into train/val
    np.random.seed(8964)
    subfolders = os.listdir(args.dump_root)
    with open(os.path.join(args.dump_root, 'train.txt'), 'w') as tf:
        with open(os.path.join(args.dump_root, 'val.txt'), 'w') as vf:
            for s in subfolders:
                if not os.path.isdir(args.dump_root + '/%s' % s):
                    continue
                imfiles = glob(os.path.join(args.dump_root, s, '*.jpg'))
                frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
                for frame in frame_ids:
                    if np.random.random() < 0.1:
                        vf.write('%s %s\n' % (s, frame))
                    else:
                        tf.write('%s %s\n' % (s, frame))
                        
if __name__ == '__main__':
    main()
