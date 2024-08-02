#!/usr/bin/env bash
#SBATCH --job-name="var_seq_interpret"
#SBATCH --output=outputs/var_seq_interpret-%j.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --cpus-per-task 1
#SBATCH --array 1-6  # https://www.memphis.edu/hpc/batchscripts.php
#SBATCH --mem=16GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

conda deactivate
conda activate ml

model=iTransformer
explainers=("winIT" "wtsr" "tsr")

function interpret {
  let index=$1-1
  let dataset=$index/2
  let seq_len_no=$index%2 

  if [ $dataset = 2 ] && [ $seq_len_no = 1 ]
  then
    seq_len=36
  else
    if [ $seq_len_no = 0 ]
    then 
        seq_len=24
    else
        seq_len=48
    fi
  fi

  echo "Running $1-th job $SLURM_JOB_ID, dataset serial $dataset, seq_len $seq_len"
  if [ $dataset = 0 ]
  then 
    echo "Running electricity with seq_len $seq_len"
    python interpret.py \
      --task_name long_term_forecast \
      --explainers $explainers\
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model $model \
      --features S \
      --seq_len $seq_len \
      --label_len 12 \
      --pred_len 24 \
      --n_features 1 --disable_progress --overwrite
    
  elif [ $dataset = 1 ]
  then
    echo "Running traffic with seq_len $seq_len"
    python interpret.py \
      --task_name long_term_forecast \
      --explainers $explainers\
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model $model \
      --features S \
      --seq_len $seq_len \
      --label_len 12 \
      --pred_len 24 \
      --n_features 1 --disable_progress --overwrite
  else
    echo "Running MIMIC with seq_len $seq_len"
    python interpret.py \
      --explainers $explainers \
      --task_name classification \
      --data mimic \
      --root_path ./dataset/mimic_iii/ \
      --data_path mimic_iii.pkl \
      --metrics auc accuracy cross_entropy \
      --model $model --n_features 31 \
      --seq_len $seq_len --disable_progress --overwrite
  fi
}

interpret $SLURM_ARRAY_TASK_ID