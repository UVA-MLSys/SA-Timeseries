python -u run.py \
  --task_name classification \
  --data mimic \
  --train \
  --result_path scratch \
  --use_gpu \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model DLinear

python -u run.py \
  --task_name classification \
  --data mimic \
  --result_path scratch \
  --use_gpu \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model DLinear

python interpret.py \
  --explainer feature_ablation occlusion augmented_occlusion feature_permutation\
  --task_name classification \
  --data mimic \
  --result_path scratch \
  --use_gpu \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics 'accuracy' 'comprehensiveness' 'sufficiency' 'log_odds' 'cross_entropy' \
  --model DLinear 
