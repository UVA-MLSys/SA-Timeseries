#!/usr/bin/env bash
#SBATCH --job-name="test"
#SBATCH --output=outputs/test.out
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=ds--6013
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

# this is for when you are using singularity
module load cuda cudnn singularity
singularity run --nv tft_pytorch.sif python test_tft.py

# module load cuda-toolkit cudnn anaconda3

# conda deactivate
# conda activate ml

# python inference.py