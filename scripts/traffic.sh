python -u run.py \
  --train \
  --use_gpu \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model DLinear \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --n_features 1 \

python -u run.py \
  --train \
  --use_gpu \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model MICN \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --n_features 1 \
  --conv_kernel 48 48

python interpret.py \
  --explainer feature_ablation occlusion augmented_occlusion feature_permutation \
  --use_gpu \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model DLinear \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --n_features 1 \

python interpret.py \
  --explainer feature_ablation occlusion augmented_occlusion feature_permutation \
  --use_gpu \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model MICN \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --n_features 1 \
  --conv_kernel 48 48