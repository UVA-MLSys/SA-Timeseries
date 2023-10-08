python -u run.py \
  --task_name long_term_forecast \
  --train \
  --result_path scratch \
  --use_gpu \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model DLinear \
  --features MS \
  --seq_len 36 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 7

python -u run.py \
  --task_name long_term_forecast \
  --train \
  --use_gpu \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model MICN \
  --features MS \
  --seq_len 36 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 7 \
  --conv_kernel 18 12

# feature_ablation occlusion augmented_occlusion gradient_shap deep_lift integrated_gradients lime 
# integrated_gradients returned out of memory error for Autoformer
# gradient_shap, deep_lift sometimes faces " One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior." error
python interpret.py \
  --explainer feature_ablation occlusion augmented_occlusion feature_permutation\
  --result_path scratch \
  --task_name long_term_forecast \
  --use_gpu \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model DLinear \
  --features MS \
  --seq_len 36 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 7 \
  --tsr