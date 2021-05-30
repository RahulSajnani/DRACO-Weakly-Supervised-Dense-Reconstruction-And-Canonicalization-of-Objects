#!/bin/bash
#BATCH -A $USER
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=0-20:00:00
#SBATCH --mail-type=END

#module add cuda/9.2
source ~/anaconda3/bin/activate
conda activate SS_NOCS
cd ~/SS_NOCS/Self-supervised-NOCS/depth-network
git checkout bug_corrections

python train.py --dataset ../data/prepare 
