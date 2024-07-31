#!/usr/bin/env bash
#SBATCH --job-name="mimic"
#SBATCH --output=scripts/outputs/mimic2.out
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --account=bii_dsc_community
# -- SBATCH --gres=gpu:v100:1
# https://www.rc.virginia.edu/userinfo/hpc/overview/#hardware-configuration
#SBATCH --gres=gpu:a100:1
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
# python interpret.py \
#   --explainers feature_ablation occlusion augmented_occlusion winIT tsr \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model DLinear \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --overwrite --disable_progress
  
# python interpret.py \
#   --explainers feature_ablation occlusion augmented_occlusion winIT tsr \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model SegRNN \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --overwrite --disable_progress
  
  
# python interpret.py \
#   --explainers feature_ablation occlusion augmented_occlusion winIT tsr \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model MICN \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --overwrite --disable_progress

# python interpret.py \
#   --explainers feature_ablation occlusion augmented_occlusion winIT tsr \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model Crossformer \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --overwrite --disable_progress
  
# python interpret.py \
#   --explainers feature_ablation occlusion augmented_occlusion winIT tsr \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model DLinear \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --overwrite --disable_progress
  
# python interpret.py \
#   --explainers feature_ablation occlusion augmented_occlusion winIT tsr \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model SegRNN \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --overwrite --disable_progress
  
  
# python interpret.py \
#   --explainers feature_ablation occlusion augmented_occlusion winIT tsr \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model MICN \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --overwrite --disable_progress

# python interpret.py \
#   --explainers feature_ablation occlusion augmented_occlusion winIT tsr \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model Crossformer \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --overwrite --disable_progress

python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion winIT tsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model DLinear --n_features 31 --seq_len 48 --overwrite --disable_progress

python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion winIT tsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model SegRNN --n_features 31 --seq_len 48 --overwrite --disable_progress

python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion winIT tsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model MICN --n_features 31 --seq_len 48 --overwrite --disable_progress

python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion winIT tsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model Crossformer --n_features 31 --seq_len 48 --overwrite --disable_progress

python interpret.py \
  --result_path scratch \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model DLinear --n_features 31 --seq_len 48 --explainers wtsr