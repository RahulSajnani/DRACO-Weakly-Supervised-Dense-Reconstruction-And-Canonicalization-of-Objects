#!/bin/bash
#BATCH -A $USER
#SBATCH -n 30
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END

#module add cuda/10.2
#module add cudnn/7.6.5-cuda-10.2
source ~/anaconda3/bin/activate
conda activate SS_NOCS
cd ~/Project/DRACO_VO/DRACO/
#git checkout refactor

#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1

#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py dataset=dataset_blender_planes
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
