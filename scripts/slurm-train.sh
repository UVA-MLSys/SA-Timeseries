#!/usr/bin/env bash
#SBATCH --job-name="mimic_DLinear_tsr"
#SBATCH --output=scripts/outputs/mimic_DLinear_tsr.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=24GB

source /etc/profile.d/modules.sh
source ~/.bashrc

# this is for when you are using singularity
module load cuda cudnn 
# module load singularity
# singularity run --nv timeseries.sif python run.py

# this is for when you have a working virtual env
conda deactivate
conda activate ml

# # replace the computing id `mi3se`` and venv name `ml` with your own
# # if you face the library linking error for anaconda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mi3se/.conda/envs/ml/lib
python interpret.py \
  --explainer feature_ablation occlusion augmented_occlusion feature_permutation\
  --task_name classification \
  --data mimic \
  --use_gpu \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --areas 0.05 0.075 0.1 0.15 \
  --metrics auc 'accuracy' 'cross_entropy' \
  --model DLinear --tsr