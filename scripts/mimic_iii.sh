python -u run.py \
  --task_name classification \
  --data mimic \
  --train \
  --use_gpu \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model LSTM

python -u run.py \
  --task_name classification \
  --data mimic \
  --train \
  --use_gpu \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model MICN \
  --conv_kernel 24 24

python run.py \
  --task_name classification \
  --data mimic \
  --result_path scratch \
  --use_gpu \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model LSTM

# feature_ablation occlusion augmented_occlusion feature_permutation
# deep_lift gradient_shap integrated_gradients -- only for transformer models
python interpret.py \
  --explainer feature_ablation occlusion augmented_occlusion feature_permutation\
  --task_name classification \
  --data mimic \
  --use_gpu \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc 'accuracy' 'cross_entropy' \
  --model LSTM 

python interpret.py \
  --explainer feature_ablation occlusion augmented_occlusion feature_permutation\
  --task_name classification \
  --data mimic \
  --use_gpu \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc 'accuracy' 'cross_entropy' \
  --model MICN \
  --conv_kernel 24 24