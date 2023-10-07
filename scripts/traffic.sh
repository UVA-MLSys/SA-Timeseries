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
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1

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
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --conv_kernel 48 48

python interpret.py \
  --explainer lime occlusion \
  --use_gpu \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model DLinear \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1