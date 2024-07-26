#!/usr/bin/env bash
#SBATCH --job-name="tsr2_traffic"
#SBATCH --output=outputs/tsr2_traffic_2.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mail-type=end
#SBATCH --mail-user=mi3se@virginia.edu
#SBATCH --mem=16GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

conda deactivate
conda activate ml

# python interpret.py \
#   --explainers tsr2 \
#   --task_name classification \
#   --data mimic \
#   --root_path ./dataset/mimic_iii/ \
#   --data_path mimic_iii.pkl \
#   --metrics auc accuracy cross_entropy \
#   --model DLinear --n_features 31 --disable_progress

# python interpret.py \
#   --explainers tsr2 \
#   --task_name classification \
#   --data mimic \
#   --root_path ./dataset/mimic_iii/ \
#   --data_path mimic_iii.pkl \
#   --metrics auc accuracy cross_entropy \
#   --model MICN --n_features 31 --disable_progress

# python interpret.py \
#   --explainers tsr2 \
#   --task_name classification \
#   --data mimic \
#   --root_path ./dataset/mimic_iii/ \
#   --data_path mimic_iii.pkl \
#   --metrics auc accuracy cross_entropy \
#   --model SegRNN --n_features 31 --disable_progress

# python interpret.py \
#   --explainers tsr2 \
#   --task_name classification \
#   --data mimic \
#   --root_path ./dataset/mimic_iii/ \
#   --data_path mimic_iii.pkl \
#   --metrics auc accuracy cross_entropy \
#   --model iTransformer --n_features 31 --disable_progress

python interpret.py \
  --task_name long_term_forecast \
  --explainers tsr2\
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model DLinear \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --disable_progress

python interpret.py \
  --task_name long_term_forecast \
  --explainers tsr2\
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model MICN \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --disable_progress

python interpret.py \
  --task_name long_term_forecast \
  --explainers tsr2\
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model SegRNN \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --disable_progress

python interpret.py \
  --task_name long_term_forecast \
  --explainers tsr2\
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model iTransformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --disable_progress

# python interpret.py \
#   --task_name long_term_forecast \
#   --explainers tsr2\
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model DLinear \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --disable_progress

# python interpret.py \
#   --task_name long_term_forecast \
#   --explainers tsr2\
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model MICN \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --disable_progress

# python interpret.py \
#   --task_name long_term_forecast \
#   --explainers tsr2\
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model SegRNN \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --disable_progress

# python interpret.py \
#   --task_name long_term_forecast \
#   --explainers tsr2\
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model iTransformer \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --disable_progress