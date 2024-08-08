#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --output=outputs/extremal_mask.out
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mem=16GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

conda deactivate
conda activate ml

models=("DLinear" "MICN" "SegRNN" "iTransformer")

function interpret {
  let index=$1-1
  model=${models[$index%4]}
  let dataset=$index/4

  echo "Running $1-th job dataset serial $dataset and model $model"
  if [ $dataset = 0 ]
  then 
    echo "Running electricity with $model"
    python interpret.py \
      --task_name long_term_forecast \
      --explainers extremal_mask\
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model $model \
      --features S \
      --seq_len 96 \
      --label_len 12 \
      --pred_len 24 \
      --n_features 1 --disable_progress
    
  elif [ $dataset = 1 ]
  then
    echo "Running traffic with $model"
    python interpret.py \
      --task_name long_term_forecast \
      --explainers extremal_mask\
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model $model \
      --features S \
      --seq_len 96 \
      --label_len 12 \
      --pred_len 24 \
      --n_features 1 --disable_progress
  else
    echo "Running MIMIC $model"
    python interpret.py \
      --explainers  extremal_mask \
      --task_name classification \
      --data mimic \
      --root_path ./dataset/mimic_iii/ \
      --data_path mimic_iii.pkl \
      --metrics auc accuracy cross_entropy \
      --model $model --n_features 31 \
      --seq_len 48 --disable_progress
  fi
}

# interpret $SLURM_ARRAY_TASK_ID

max=12
for (( i=1; i <= $max; ++i ))
do
    interpret $i
done