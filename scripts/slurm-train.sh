#!/usr/bin/env bash
#SBATCH --job-name="total"
#SBATCH --output=train.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=24GB

source /etc/profile.d/modules.sh
source ~/.bashrc

# this is for when you are using singularity
module load cuda cudnn singularity
singularity run --nv timeseries.sif python run.py

# singularity run --nv timeseries.sif python run.py

# # this is for when you have a working virtual env
# module load cuda cudnn anaconda

# conda deactivate
# conda activate ml

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mi3se/.conda/envs/ml/lib
# python run.py