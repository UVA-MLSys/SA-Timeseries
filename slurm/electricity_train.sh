#!/usr/bin/env bash
#SBATCH --job-name="electricity_train"
#SBATCH --output=outputs/electricity_train.out
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=mi3se@virginia.edu
#SBATCH --mem=24GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

conda deactivate
conda activate ml

python run.py \
  --task_name long_term_forecast \
  --train \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --model DLinear 

python run.py \
  --task_name long_term_forecast \
  --train \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --model SegRNN

python run.py \
  --task_name long_term_forecast \
  --train \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --model MICN

python run.py \
  --task_name long_term_forecast \
  --train \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --model Crossformer 

python run.py \
  --task_name long_term_forecast \
  --train \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --model TimesNet   