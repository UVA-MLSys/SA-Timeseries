python run.py \
  --task_name long_term_forecast \
  --train \
  --result_path scratch \
  --use_gpu \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model DLinear \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1

  python run.py \
  --task_name long_term_forecast \
  --train \
  --result_path scratch \
  --use_gpu \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model Autoformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1

  python interpret.py \
  --task_name long_term_forecast \
  --explainer lime occlusion \
  --train \
  --result_path scratch \
  --use_gpu \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model DLinear \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1