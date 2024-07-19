#!/usr/bin/env bash
#SBATCH --job-name="traffic_interpret"
#SBATCH --output=outputs/traffic_interpret.out
#SBATCH --partition=gpu
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mail-type=end
#SBATCH --mail-user=mi3se@virginia.edu
#SBATCH --mem=24GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

conda deactivate
conda activate ml

python interpret.py \
  --task_name long_term_forecast \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT tsr wtsr\
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model DLinear \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --overwrite --disable_progress

python interpret.py \
  --task_name long_term_forecast \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT tsr wtsr\
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model MICN \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --overwrite --disable_progress

python interpret.py \
  --task_name long_term_forecast \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT tsr wtsr\
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model SegRNN \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --overwrite --disable_progress

python interpret.py \
  --task_name long_term_forecast \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT tsr wtsr\
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model Crossformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --overwrite --disable_progress

python interpret.py \
  --task_name long_term_forecast \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT tsr wtsr\
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model TimesNet \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --overwrite --disable_progress