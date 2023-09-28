from run import *
import tint
from tint.metrics import mae, mse
import os, gc
import pandas as pd
from tqdm import tqdm
from utils.explainer import *
from exp.exp_interpret import Exp_Interpret

from captum.attr import (
    DeepLift,
    GradientShap,
    IntegratedGradients,
    Lime
)

from tint.attr import (
    AugmentedOcclusion,
    DynaMask,
    Occlusion, 
    FeatureAblation
)

parser = get_parser()
argv = """
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
""".split()
args = parser.parse_args(argv)

explainer_name_map = {
    "deep_lift":DeepLift,
    "gradient_shap":GradientShap,
    "integrated_gradients":IntegratedGradients,
    "lime":Lime,
    "occlusion":Occlusion,
    # "augmented_occlusion":AugmentedOcclusion, requires data when initializing
    # "dyna_mask":DynaMask,
    "feature_ablation":FeatureAblation
}

set_random_seed(args.seed)
# Disable cudnn if using cuda accelerator throws error.
# Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
# args.use_gpu = False
    
assert args.task_name == 'long_term_forecast', "Only long_term_forecast is supported for now"

Exp = Exp_Long_Term_Forecast
    
setting = stringify_setting(args)
exp = Exp(args)  # set experiments
flag = 'test'
_, dataloader = exp._get_data(flag)

exp.model.load_state_dict(
    torch.load(os.path.join('checkpoints/' + setting, 'checkpoint.pth'))
)
result_folder = './results/' + setting + '/'

# explainers = ['deep_lift', 'gradient_shap', 'integrated_gradients', 'lime', 'feature_ablation']
explainers = ['feature_ablation']
areas = [0.01, 0.02, 0.05, 0.95, 0.99]
interpreter = Exp_Interpret(
    exp.model, result_folder, exp.device, args, 
    explainers, explainer_name_map, areas
) 

interpreter.interpret(dataloader, flag, baseline_mode='zeros')
interpreter.interpret(dataloader, flag, tsr=True, baseline_mode='zeros')