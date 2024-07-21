#!/usr/bin/env bash
#SBATCH --job-name="electricity_interpret"
#SBATCH --output=outputs/electricity_interpret_2.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
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
#   --task_name long_term_forecast \
#   --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT wtsr tsr\
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model DLinear \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --overwrite --disable_progress

# python interpret.py \
#   --task_name long_term_forecast \
#   --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT wtsr tsr\
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model MICN \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --overwrite --disable_progress

# python interpret.py \
#   --task_name long_term_forecast \
#   --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT wtsr tsr\
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model SegRNN \
#   --features S \
#   --seq_len 96 \
#   --label_len 12 \
#   --pred_len 24 \
#   --n_features 1 --overwrite --disable_progress

python interpret.py \
  --task_name long_term_forecast \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT wtsr tsr\
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model Crossformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --overwrite --disable_progress

python interpret.py \
  --task_name long_term_forecast \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT wtsr tsr\
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model TimesNet \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --overwrite --disable_progress