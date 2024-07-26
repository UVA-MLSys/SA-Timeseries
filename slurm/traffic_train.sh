#!/usr/bin/env bash
#SBATCH --job-name="traffic_train"
#SBATCH --output=outputs/traffic_train.out
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

models=("DLinear" "MICN" "SegRNN" "iTransformer")

for model in ${models[@]}
do 
echo "Running for model:$model"
python run.py \
  --task_name long_term_forecast \
  --train \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --model $model
done