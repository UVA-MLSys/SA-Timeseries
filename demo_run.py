from run import *
import torch
argv = """
  --task_name long_term_forecast \
  --train \
  --use_gpu \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model PatchTST \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1
""".split()
parser = get_parser()
args = parser.parse_args(argv)
main(args)