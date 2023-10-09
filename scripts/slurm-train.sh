#!/usr/bin/env bash
#SBATCH --job-name="electricity_MICN_tsr"
#SBATCH --output=scripts/outputs/electricity_MICN_tsr.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=24GB

source /etc/profile.d/modules.sh
source ~/.bashrc

# this is for when you are using singularity
module load cuda cudnn 
# module load singularity
# singularity run --nv timeseries.sif python run.py

# this is for when you have a working virtual env
conda deactivate
conda activate ml

# # replace the computing id `mi3se`` and venv name `ml` with your own
# # if you face the library linking error for anaconda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mi3se/.conda/envs/ml/lib
python interpret.py \
  --explainer feature_ablation occlusion augmented_occlusion feature_permutation \
  --use_gpu \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model MICN \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --conv_kernel 18 12 \
  --tsr  