python -u run.py \
  --task_name long_term_forecast \
  --train \
  --use_gpu \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model Transformer \
  --features MS \
  --seq_len 36 \
  --label_len 12 \
  --pred_len 24 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7

python interpret.py \
  --explainer lime occlusion \
  --task_name long_term_forecast \
  --use_gpu \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model Transformer \
  --features MS \
  --seq_len 36 \
  --label_len 12 \
  --pred_len 24 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7