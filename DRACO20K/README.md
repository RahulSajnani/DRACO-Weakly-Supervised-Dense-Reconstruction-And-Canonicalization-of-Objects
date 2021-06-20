# DRACO20K

#### About

This repository contains scripts to generate DRACO20K dataset from ShapeNet models. 

#### Setup instructions

Blender is installed with its own bundled python binary. But we will need to install imageio to run the following script as it is not installed by default.

```bash
# python and blender version can be different. For older versions of blender you will have to install pip
<blender-download-path>/2.83/python/bin/python3.7m -m ensurepip
<blender-download-path>/2.83/python/bin/python3.7m -m pip install -U pip
<blender-download-path>/2.83/python/bin/python3.7m -m pip install imageio
```



Setting the blender paths and rendering parameters in settings.py. Default values are assigned.

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





#### Running the script

```bash
<blender-download-path>/blender --python NOCS_render.py -b
```



#### Version 

Blender and ShapeNet versions used by this repo:

- Blender 2.83

- ShapeNetCore.v2

