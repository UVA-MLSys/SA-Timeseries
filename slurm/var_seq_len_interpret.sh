#!/usr/bin/env bash
#SBATCH --job-name="var_seq_len_interpret"
#SBATCH --output=outputs/var_seq_len_interpret.out
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

# the following are for ablation studies
python interpret.py \
  --task_name long_term_forecast \
  --explainers winIT wtsr tsr\
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model iTransformer \
  --features S \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --disable_progress

python interpret.py \
  --task_name long_term_forecast \
  --explainers winIT wtsr tsr\
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model iTransformer \
  --features S \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --disable_progress

python interpret.py \
  --task_name long_term_forecast \
  --explainers winIT wtsr tsr\
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model iTransformer \
  --features S \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --disable_progress

python interpret.py \
  --task_name long_term_forecast \
  --explainers winIT wtsr tsr\
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model iTransformer \
  --features S \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --disable_progress

python interpret.py \
  --explainers winIT wtsr tsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model iTransformer --n_features 31 --disable_progress --seq_len 36

python interpret.py \
  --explainers winIT wtsr tsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model iTransformer --n_features 31 --disable_progress --seq_len 24