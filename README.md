# DRACO: Weakly Supervised <u>D</u>ense <u>R</u>econstruction <u>A</u>nd <u>C</u>anonicalization Of <u>O</u>bjects





![DRACO pipeline](./images/pipeline.gif)



## Abstract

We present **DRACO**, a method for **D**ense **R**econstruction **A**nd **C**anonicalization of **O**bject shape from one or more RGB images. Canonical shape reconstruction— estimating 3D object shape in a coordinate space canonicalized for scale, rotation, and translation parameters—is an emerging paradigm that holds promise for a multitude of robotic applications. Prior approaches either rely on painstakingly gathered dense 3D supervision, or produce only sparse canonical representations, limiting real-world applicability. DRACO performs dense canonicalization using only weak supervision in the form of camera poses and semantic keypoints at train time. During inference, DRACO predicts dense object-centric depth maps in a canonical coordinate-space, solely using one or more RGB images of an object. Extensive experiments on canonical shape reconstruction and pose estimation show that DRACO is competitive or superior to fully-supervised methods.



## Dataset

**Prepared** datasets are the ones prepared to train wherein we take 3 consecutive data sample from the DRACO20K dataset and club them.  



| Dataset                       | Link                                                         | Size (GB) |
| ----------------------------- | ------------------------------------------------------------ | --------- |
| Cars (**prepared**) (small)   | [link](https://iiitaphyd-my.sharepoint.com/:u:/r/personal/robotics_iiit_ac_in/Documents/DRACO/car_reflection_canonical_wo_depth.zip?csf=1&web=1&e=I2EE6N) | 3.8       |
| Planes (**prepared**) (small) | Coming soon                                                  | 3.6       |
| DRACO20K cars                 | Coming soon                                                  | 89        |
| DRACO20K planes               | Coming soon                                                  | 15        |

To begin training, download the Cars (**prepared**) dataset and save unzip to path `./data/DRACO20K_cars`. 



## Running the code

1. #### Environment

   Follow the following instructions to load the environment.

   ```bash
   conda create -f environment.yaml
   # As tk3dv is not available on PyPi this will throw an error while installing tk3dv but that is not an issue
   conda activate DRACO
   # Install tk3dv manually in the same environment
   pip install git+https://github.com/drsrinathsridhar/tk3dv.git
   ```

2. #### Training

   Please refer to the configuration file in `./DRACO/cfgs/config_DRACO.yaml` and change the path to the dataset and set the hyper-parameters.

   **Note:** As the second training phase is heavy (due to the DRACO + VGG (perceptual loss) + multi-view consistency), make sure you set the batch_size as the number of GPUs available for training. For instance, if you have 2 GPUs set batch_size to 2 and accumulated_num_batches to 6 (2 x 6 = 12)

   ```bash
   cd DRACO
   # Before running the script change the path to the dataset in /DRACO/cfgs/config_DRACO.yaml
   
   CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
   ```

   

3. #### Testing

   ```bash
   cd DRACO
   # For DRACO20K dataset
   CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation.py --model <path_to_checkpoint> --path <path_to_directory_with_images> --output <path_to_output_directory> --real 0
   
   # For Real dataset
   CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation.py --model <path_to_checkpoint> --path <path_to_directory_with_images> --output <path_to_output_directory> --real 1
   ```

   



## Pretrained models

Coming soon.



## Citation



If you find our work helpful, please consider to citing:

```
@misc{sajnani2020draco,
title={DRACO: Weakly Supervised Dense Reconstruction And Canonicalization of Objects}, 
author={Rahul Sajnani and AadilMehdi Sanchawala and Krishna Murthy Jatavallabhula and Srinath Sridhar and K. Madhava Krishna},
year={2020},
eprint={2011.12912},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```

