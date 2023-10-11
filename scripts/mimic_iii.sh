python -u run.py \
  --task_name classification \
  --data mimic \
  --train \
  --result_path scratch \
  --use_gpu \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model DLinear \
  --features MS \
  --seq_len 48 \
  --n_features 31

python -u run.py \
  --task_name classification \
  --data mimic \
  --result_path scratch \
  --use_gpu \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model DLinear