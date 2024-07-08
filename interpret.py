from run import initial_setup, get_parser as get_main_parser, set_random_seed
import os, json
from typing import Union
from exp.exp_long_term_forecasting import *
from exp.exp_classification import Exp_Classification
from utils.explainer import *
from exp.exp_interpret import Exp_Interpret, explainer_name_map

def main(args):
    initial_setup(args)
    print(args)

    # Disable cudnn if using cuda accelerator throws error.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # args.use_gpu = False
        
    if args.task_name == 'classification': Exp = Exp_Classification
    else: Exp = Exp_Long_Term_Forecast
    
    parent_seed = args.seed
    np.random.seed(parent_seed)
    experiment_seeds = np.random.randint(1e3, size=args.itrs)
    original_itr = args.itr_no
    
    for itr_no in range(1, args.itrs+1):
        if (original_itr is not None) and original_itr != itr_no: continue
        
        args.seed = experiment_seeds[itr_no-1]
        print(f'\n>>>> itr_no: {itr_no}, seed: {args.seed} <<<<<<')
        set_random_seed(args.seed)
        args.itr_no = itr_no
        
        exp = Exp(args)  # set experiments
        _, dataloader = exp._get_data(args.flag)

        exp.load_best_model()

        # Some models don't work with gradient based explainers
        # explainers = ['deep_lift', 'gradient_shap', 'integrated_gradients']

        interpreter = Exp_Interpret(exp, dataloader) 
        interpreter.interpret(dataloader)
        print()
        
    args.seed = parent_seed
    config_filepath = os.path.join(args.result_path, stringify_setting(args), 'config_interpret.json')
    args.seeds = [int(seed) for seed in experiment_seeds]
    with open(config_filepath, 'w') as output_file:
        json.dump(vars(args), output_file, indent=4)
    
def get_parser():
    parser = get_main_parser()
    parser.description = 'Interpret timeseries model'
    parser.add_argument('--explainers', nargs='+', default=['feature_ablation'], 
        choices=list(explainer_name_map.keys()),
        help='explaination method names. Gradient based explainers are not supported yet for regression')
    parser.add_argument('--areas', nargs='*', type=float, default=[0.05, 0.075, 0.1, 0.15],
        help='top k features to keep or mask during evaluation')
    parser.add_argument('--baseline_mode', type=str, default='random',
        choices=['random', 'aug', 'zero', 'mean'],
        help='how to create the baselines for the interepretation methods')
    parser.add_argument('--metrics', nargs='*', type=str, default=['mae', 'mse'], 
        help='interpretation evaluation metrics')
    parser.add_argument('--overwrite', action='store_true', help='overwrite previous results')
    parser.add_argument('--dump_attrs', action='store_true', help='dump raw attributes in torch file')
    
    parser.add_argument('--disable_progress', action='store_true', help='disble progress bar')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)