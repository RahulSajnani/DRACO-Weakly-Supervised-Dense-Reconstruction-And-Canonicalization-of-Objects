# DRACO: Weakly Supervised <u>D</u>ense <u>R</u>econstruction <u>A</u>nd <u>C</u>anonicalization Of <u>O</u>bjects





![DRACO pipeline](./images/pipeline.gif)



## Abstract

We present **DRACO**, a method for **<u>D</u>**ense **<u>R</u>**econstruction **<u>A</u>**nd <u>**C**</u>anonicalization of **<u>O</u>**bject shape from one or more RGB images. Canonical shape reconstruction— estimating 3D object shape in a coordinate space canonicalized for scale, rotation, and translation parameters—is an emerging paradigm that holds promise for a multitude of robotic applications. Prior approaches either rely on painstakingly gathered dense 3D supervision, or produce only sparse canonical representations, limiting real-world applicability. DRACO performs dense canonicalization using only weak supervision in the form of camera poses and semantic keypoints at train time. During inference, DRACO predicts dense object-centric depth maps in a canonical coordinate-space, solely using one or more RGB images of an object. Extensive experiments on canonical shape reconstruction and pose estimation show that DRACO is competitive or superior to fully-supervised methods.



## Running the code

1. #### Environment

   Follow the following instructions to load the environment.

   ```bash
   conda create -f environment.yaml
   conda activate DRACO
   pip install git+https://github.com/drsrinathsridhar/tk3dv.git
   ```

2. #### Training

   ```bash
   cd DRACO
   CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
   ```

   

3. #### Testing

   ```bash
   cd DRACO
   CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation.py 
   ```

   

## Dataset

We have zipped a smaller version of our training cars dataset. Download our dataset [here]().



## Pretrained models



## Citation

Cite us, if you find our work helpful.

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

