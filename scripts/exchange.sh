python -u run.py \
  --train \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model Transformer \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --n_feature 8