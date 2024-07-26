#!/usr/bin/env bash
#SBATCH --job-name="traffic_interpret"
#SBATCH --output=outputs/traffic_interpret.out
#SBATCH --partition=gpu
#SBATCH --time=16:00:00
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

explainers=("feature_ablation" "occlusion" "augmented_occlusion" "feature_permutation" "integrated_gradients" "gradient_shap" "dyna_mask" "winIT" "wtsr" "tsr")
models=("DLinear" "MICN" "SegRNN" "iTransformer")

for model in ${models[@]}
do 
echo "Running for model:$model"

python interpret.py \
  --task_name long_term_forecast \
  --explainers ${explainers[@]}\
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model $model \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --disable_progress --overwrite

done