#!/usr/bin/env bash
#SBATCH --job-name="traffic_Autoformer_tsr"
#SBATCH --output=scripts/outputs/traffic_Autoformer_tsr.out
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
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
# module load anaconda
# conda init
conda deactivate
conda activate ml

# # replace the computing id `mi3se`` and venv name `ml` with your own
# # if you face the library linking error for anaconda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mi3se/.conda/envs/ml/lib
python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model Autoformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 