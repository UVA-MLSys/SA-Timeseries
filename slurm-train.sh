#!/usr/bin/env bash
#SBATCH --job-name="total"
#SBATCH --output=outputs/train.out
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --account=ds--6013
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=24GB

source /etc/profile.d/modules.sh
source ~/.bashrc

# this is for when you are using singularity
module load cuda cudnn singularity
singularity run --nv tft_pytorch.sif python train_tft.py

# singularity run --nv tft_pytorch.sif python train_tft.py --input-file=2022_May_age_groups/Total.csv --output=tft_pytorch/scratch/total

# # this is for when you have a working virtual env
# module load cuda cudnn anaconda

# conda deactivate
# conda activate ml

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mi3se/.conda/envs/ml/lib
# python train.py