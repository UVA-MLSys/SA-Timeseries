#!/usr/bin/env bash
#SBATCH --job-name="mimic_interpret"
#SBATCH --output=outputs/mimic_interpret.out
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
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT wtsr tsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model DLinear --n_features 31 --overwrite --disable_progress

python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT wtsr tsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model MICN --n_features 31 --overwrite --disable_progress

python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT wtsr tsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model SegRNN --n_features 31 --overwrite --disable_progress

python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT wtsr tsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model Crossformer --n_features 31 --overwrite --disable_progress

python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation winIT wtsr tsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model TimesNet --n_features 31 --overwrite --disable_progress