from run import initial_setup, get_parser as get_main_parser
import os
from typing import Union
from exp.exp_long_term_forecasting import *
from exp.exp_classification import Exp_Classification
from utils.explainer import *
from exp.exp_interpret import Exp_Interpret, explainer_name_map

def main(args):
    initial_setup(args)

    # Disable cudnn if using cuda accelerator throws error.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # args.use_gpu = False
        
    if args.task_name == 'classification': Exp = Exp_Classification
    else: Exp = Exp_Long_Term_Forecast
    exp = Exp(args)  # set experiments
    _, dataloader = exp._get_data(args.flag)

    exp.load_best_model()

    # Some models don't work with gradient based explainers
    # explainers = ['lime', 'feature_ablation', 'deep_lift', 'gradient_shap', 'integrated_gradients']

    interpreter = Exp_Interpret(exp, dataloader) 
    interpreter.interpret(dataloader)
    
def get_parser():
    parser = get_main_parser()
    parser.description = 'Interpret timeseries model'
    parser.add_argument('--tsr', action='store_true', help='Run interpretation methods with TSR enabled')
    parser.add_argument('--explainers', nargs='*', default=['feature_ablation'], 
        choices=list(explainer_name_map.keys()),
        help='explaination method names')
    parser.add_argument('--areas', nargs='*', type=float, default=[0.05, 0.075, 0.1, 0.15],
        help='top k features to keep or mask during evaluation')
    parser.add_argument('--baseline_mode', type=str, default='random',
        choices=['random', 'aug', 'zero', 'mean'],
        help='how to create the baselines for the interepretation methods')
    parser.add_argument('--metrics', nargs='*', type=str, default=['mae', 'mse'], 
        help='interpretation evaluation metrics')
    parser.add_argument('--threshold', type=float, default=0, 
        help='Threshold for the feature step computation')
    parser.add_argument(
        '--attr_by_pred', action='store_true', 
        help='evaluate the attr by each predicted class/horizon. Otherwise take the average over the output horizon.')
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)