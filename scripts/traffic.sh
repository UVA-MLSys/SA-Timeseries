python -u run.py \
  --train \
  --use_gpu \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model DLinear \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \

python -u run.py \
  --train \
  --use_gpu \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model MICN \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --conv_kernel 18 12

python interpret.py \
  --explainer feature_ablation occlusion augmented_occlusion feature_permutation \
  --use_gpu \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model DLinear \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \

# feature_ablation occlusion augmented_occlusion feature_permutation
# deep_lift gradient_shap integrated_gradients -- only for transformer models
python interpret.py \
  --explainer winIT \
  --use_gpu \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model MICN \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --conv_kernel 18 12