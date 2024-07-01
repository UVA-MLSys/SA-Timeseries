python -u run.py \
  --task_name long_term_forecast \
  --train \
  --result_path scratch \
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
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model MICN \
  --features MS \
  --seq_len 36 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 7 \
  --conv_kernel 18 12

# perturbation based: feature_ablation occlusion augmented_occlusion feature_permutation
# gradient based methods didn't work for DLinear and MICN
# grad based: gradient_shap deep_lift integrated_gradients lime 
# integrated_gradients returned out of memory error for Autoformer
# gradient_shap, deep_lift sometimes faces " One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior." error
python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation\
  --task_name long_term_forecast \
  --result_path scratch \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model DLinear \
  --features MS \
  --seq_len 36 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 7