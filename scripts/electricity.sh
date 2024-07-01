python run.py \
  --task_name long_term_forecast \
  --train \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --model DLinear 

python run.py \
  --task_name long_term_forecast \
  --train \
  --result_path scratch \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model Crossformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1

python run.py \
  --task_name long_term_forecast \
  --train \
  --result_path scratch \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model MICN \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --conv_kernel 18 12

# feature_ablation occlusion augmented_occlusion feature_permutation
# deep_lift gradient_shap integrated_gradients -- only for transformer models
python interpret.py \
  --task_name long_term_forecast \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation\
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model Crossformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1

python interpret.py \
  --task_name long_term_forecast \
  --explainers feature_ablation \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model DLinear \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --result_path scratch \
  