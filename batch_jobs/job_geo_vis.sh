#!/bin/bash
#BATCH -A $USER
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END

#module add cuda/9.2
source ~/anaconda3/bin/activate
conda activate SS_NOCS
cd ~/SS_NOCS/Self-supervised-NOCS/depth-network
git checkout bug_corrections

python train_geometric_vis.py --dataset ../data/prepare --learning_rate 1e-5 
