# DRACO20K

This repository contains scripts to generate DRACO20K dataset from ShapeNet models. 



## Dataset structure

This section details the dataset structure (actual and prepared) used by DRACO and some additional details.

### DRACO20K

The following is the DRACO20K dataset structure:

```bash
├── DRACO20K
    ├── a2d1b78e03f3cc39d1e95557cb698cdf   - Training sequence ID
        ├── frame_00000000_CameraPose.json - Camera poses                - json
        ├── frame_00000000_Color_00.png    - RGB single view image       - H x W x 3 
        ├── frame_00000000_Depth_00.exr    - Ground truth Depth maps     - H x W x 1 
        ├── frame_00000000_KeyPoints.npy   - Keypoints with visibility   - N x 3 	
        ├── frame_00000000_Mask_00.png     - Ground truth Masks			 - H x W x 1 
        ├── frame_00000000_NOXRayTL_00.png - NOCS Ground truth images    - H x W x 3
```

**Note**: In addition to the above files we also pre-train C3DPO on our dataset and store the transformations that transform the object from the current frame of reference to the canonical frame of reference.



### Prepared data

The prepared dataset combines data samples in groups of 3 for training the multi view setting. The preparation script is available in `<repo>/data/prepare_train_data.py`

Prepared dataset structure:

```bash
├── DRACO20K_prepared
    ├── a2d1b78e03f3cc39d1e95557cb698cdf 
        ├── CameraPose_00000001.json                              - Camera poses 
        ├── keypoints_00000001.npy  	                          - Keypoints         			              - V x 1   x 3 x N
        ├── view_00000001.jpg              						  - RGB images		   			              - H x V*W x 3
        ├── c3dpo_00000001_rotation.npy    						  - Canonical to view rotation (from C3DPO) - V x 1   x 3 x 3
        ├── depth_00000001.tiff (not used in training)            - Ground truth depth maps                  - H x V*W x 1      
        ├── mask_00000001.jpg                                     - Ground truth masks
        			  - H x V*W x 1
        ├── c3dpo_00000001_cam_coords.npy	                      - C3DPO lifted keypoints
        			  - V x 1   x 3 x N
       	├── nocs_00000001.jpg   (not used in training)            - Ground truth NOCS maps

# Notations for dimensions
# V = Number of views 
# W = Width of image
# H = Height of image
# N = Number of keypoints
```



## Setup instructions

Blender is installed with its own bundled python binary. But we will need to install imageio to run the following script as it is not installed by default.

```bash
# python and blender version can be different. For older versions of blender you will have to install pip
<blender-download-path>/2.83/python/bin/python3.7m -m ensurepip
<blender-download-path>/2.83/python/bin/python3.7m -m pip install -U pip
<blender-download-path>/2.83/python/bin/python3.7m -m pip install imageio
```

Setting the blender paths and rendering parameters in `settings.py`. Default values are assigned.

```python
# Shapenet path
g_shapenet_path = './ShapeNetCore.v2'
# Output folder path
g_output_path = './outputs'
# Background folder location
g_background_image_path = "./background_image"
# Number of images per model
g_number_images = 50
# json for 3d keypoints
g_kp_json = "./car.json"
```



## Rendering the dataset and preparing for training



To render your DRACO20K dataset using Blender:

```bash
<blender-download-path>/blender --python NOCS_render.py -b
```



#### Preparing the DRACO20K dataset for training

```bash
cd <repo>/data/

python prepare_train_data.py --dataset_name cars_blender_nocs --dataset_dir <DRACO20K-complete-dataset-path> --canonical <canonical,set=1> --num_threads <number-of-threads> --seq_length <length-of-views,set=3> --dump_root <output-path>
```



### Additional details

#### Blender coordinate system

Blender 2.83 uses a right handed coordinate frame wherein the X-axis is towards the right, Y-axis is up, and the Z-axis is backward (inwards into the camera). In our Data loader, we transform this coordinate frame to the CV coordinate frame X-right, Y-down, and Z-forward. 

#### Camera intrinsics

Camera intrinsic matrix used in the rendering script.

```python
K = np.array([[888.88,    0,   320],
              [     0, 1000,   240],
              [     0,    0,     1]])
```



#### Version 

Blender and ShapeNet versions used by this repo:

- Blender 2.83

- ShapeNetCore.v2

