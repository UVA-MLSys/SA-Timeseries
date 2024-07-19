#!/usr/bin/env bash
#SBATCH --job-name="mimic_train"
#SBATCH --output=outputs/mimic_train.out
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
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model DLinear --n_features 31

python run.py \
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model SegRNN --n_features 31

python run.py \
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model MICN --n_features 31

python run.py \
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model Crossformer --n_features 31

python run.py \
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model TimesNet --n_features 31