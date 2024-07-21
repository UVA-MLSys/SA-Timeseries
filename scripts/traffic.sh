python run.py \
  --train \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --model DLinear 

python run.py \
  --train \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model MICN \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1

python run.py \
  --train \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model SegRNN \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1

python run.py \
  --train \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model Crossformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1

python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion winIT tsr \
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
  --explainers wtsr \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model iTransformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --result_path scratch --overwrite --itr_no 1