python -u run.py \
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model DLinear

python -u run.py \
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model MICN \

python -u run.py \
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model SegRNN

python -u run.py \
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model Crossformer

# feature_ablation occlusion augmented_occlusion feature_permutation
# deep_lift gradient_shap integrated_gradients -- only for transformer models
python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion deep_lift fit winIT tsr wtsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model DLinear

python interpret.py \
  --explainers feature_ablation \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model DLinear