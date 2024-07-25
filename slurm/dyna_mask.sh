#!/usr/bin/env bash
#SBATCH --job-name="dyna_mask"
#SBATCH --output=outputs/dyna_mask2.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
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

models=("DLinear" "MICN" "SegRNN" "iTransformer" "Crossformer")
explainer=dyna_mask

for model in ${models[@]}
do
python interpret.py \
  --explainers $explainer \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model $model --n_features 31 --disable_progress

done

for model in ${models[@]}
do
python interpret.py \
  --task_name long_term_forecast \
  --explainers $explainer \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model $model \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --disable_progress

done

for model in ${models[@]}
do
python interpret.py \
  --task_name long_term_forecast \
  --explainers $explainer \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model $model \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --disable_progress

done