python -u run.py \
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model DLinear --n_features 31

python -u run.py \
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model MICN --n_features 31

python -u run.py \
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model SegRNN --n_features 31

python -u run.py \
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model Crossformer --n_features 31

# feature_ablation occlusion augmented_occlusion feature_permutation
# deep_lift gradient_shap integrated_gradients -- only for transformer models
python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion deep_lift fit winIT tsr wtsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model DLinear --n_features 31

python interpret.py \
  --explainers winIT \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model MICN --n_features 31 --overwrite --itr_no 3