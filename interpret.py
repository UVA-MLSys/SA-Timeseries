from run import initial_setup, get_parser as get_main_parser
import os
from exp.exp_long_term_forecasting import *
from utils.explainer import *
from exp.exp_interpret import Exp_Interpret, explainer_name_map

def main(args):
    initial_setup(args)

    # Disable cudnn if using cuda accelerator throws error.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # args.use_gpu = False
        
    assert args.task_name == 'long_term_forecast', "Only long_term_forecast is supported for now"

    exp = Exp_Long_Term_Forecast(args)  # set experiments
    _, dataloader = exp._get_data(args.flag)

    exp.load_best_model()

    # PatchTST doesn't work with gradient based explainers
    # explainers = ['lime', 'feature_ablation', 'deep_lift', 'gradient_shap', 'integrated_gradients']

    interpreter = Exp_Interpret(
        exp.model, exp.output_folder, exp.device, args, dataloader
    ) 

    interpreter.interpret(
        dataloader, args.flag, tsr=args.tsr, 
        baseline_mode=args.baseline_mode
    )
    
    
def get_parser():
    parser = get_main_parser()
    parser.description = 'Interpret timeseries model'
    parser.add_argument('--tsr', action='store_true', help='Run interpretation methods with TSR enabled')
    parser.add_argument('--explainers', nargs='*', default=['feature_ablation'], 
        choices=list(explainer_name_map.keys()),
        help='explaination method names')
    parser.add_argument('--areas', nargs='*', type=float, default=[0.05, 0.075, 0.1, 0.2],
        help='top k features to keep or mask during evaluation')
    parser.add_argument('--baseline_mode', type=str, default='random',
        choices=['random', 'aug', 'zero', 'mean'],
        help='how to create the baselines for the interepretation methods')
    parser.add_argument('--metrics', nargs='*', type=str, default=['mae', 'mse'], 
        help='interpretation evaluation metrics')
    parser.add_argument('--flag', type=str, default='test', choices=['train', 'val', 'test'],
        help='data split type')
    # parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)