python -u run.py --task_name long_term_forecast --train --use_gpu --result_path scratch --root_path ./dataset/illness/ --data_path national_illness.csv --model MICN --data custom --features MS --seq_len 36 --label_len 36 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 768 --d_ff 768 --top_k 5 --conv_kernel 18 12

model_name=MICN

python -u run.py \
  --task_name long_term_forecast \
  --train \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 36 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 768 \
  --d_ff 768 \
  --top_k 5 \
  --des 'Exp' \
  --conv_kernel 18 12