import os, torch
from tqdm import tqdm 
import pandas as pd
from utils.explainer import *
from utils.tsr_tunnel import *
from tint.metrics import mae, mse, accuracy, cross_entropy, lipschitz_max
from datetime import datetime

from captum.attr import (
    DeepLift,
    GradientShap,
    IntegratedGradients,
    Lime,
    FeaturePermutation
)

from tint.attr import (
    AugmentedOcclusion,
    DynaMask,
    Fit,
    Occlusion, 
    FeatureAblation
)

expl_metric_map = {
    'mae': mae, 'mse': mse, 'accuracy': accuracy, 
    'cross_entropy':cross_entropy, 'lipschitz_max': lipschitz_max
}

explainer_name_map = {
    "deep_lift":DeepLift,
    "gradient_shap":GradientShap,
    "integrated_gradients":IntegratedGradients,
    "lime":Lime,
    "occlusion":Occlusion,
    "augmented_occlusion":AugmentedOcclusion, # requires data when initializing
    "dyna_mask":DynaMask, # needs additional arguments when initializing
    "fit": Fit,
    "feature_ablation":FeatureAblation,
    "feature_permutation":FeaturePermutation
}

class Exp_Interpret:
    def __init__(
        self, model, result_folder, device, args, dataloader
    ) -> None:
        assert not args.output_attention, 'Model needs to output target only'
        
        self.args = args
        self.result_folder = result_folder
        self.device = device
        
        print(f'explainers: {args.explainers}\n areas: {args.areas}\n metrics: {args.metrics}\n')
        
        model.eval()
        model.zero_grad()
        self.model = model
        
        self.explainers_map = dict()
        for name in args.explainers:
            if name == 'augmented_occlusion':
                all_inputs = get_total_data(dataloader, self.device)
                self.explainers_map[name] = explainer_name_map[name](self.model, all_inputs)
            else:
                self.explainers_map[name] = explainer_name_map[name](self.model)
    
    def interpret(self, dataloader, flag, tsr=False, baseline_mode='random'):
        result_columns = ['batch_index', 'metric', 'area', 'comp', 'suff']
        if tsr:
            print('Interpreting with TSR enabled.')
        
        for name in self.args.explainers:
            results = []
            start = datetime.now()
            print(f'Running {name} from {start}')
            progress_bar = tqdm(
                enumerate(dataloader), total=len(dataloader), disable=False
            )
            
            for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in progress_bar:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                # outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                inputs = (batch_x, batch_x_mark)
                # baseline must be a scaler or tuple of tensors with same dimension as input
                baselines = get_baseline(inputs, mode=baseline_mode)
                additional_forward_args = (dec_inp, batch_y_mark)

                # get attributions
                batch_results = self.evaluate(
                    name, inputs, baselines, 
                    additional_forward_args, tsr, batch_index
                )
                results.extend(batch_results)
            
            end = datetime.now()
            print(f'Experiment ended at {end}. Total time taken {end - start}.')
            if tsr:
                self.dump_results(results, result_columns, f'tsr_{name}.csv')
            else:
                self.dump_results(results, result_columns, f'{name}.csv')
            
    def evaluate(
        self, name, inputs, baselines, 
        additional_forward_args, tsr, batch_index
    ):
        explainer = self.explainers_map[name]
        if tsr:
            explainer = TSRTunnel(explainer)
            if type(inputs) == tuple:
                sliding_window_shapes = tuple([(1,1) for _ in inputs])
                strides = tuple([1 for _ in inputs])
            else:
                sliding_window_shapes = (1,1)
                strides = 1
                
            attr = compute_tsr_attr(
                self.args, explainer, inputs=inputs, 
                sliding_window_shapes=sliding_window_shapes, 
                strides=strides, baselines=baselines,
                additional_forward_args=additional_forward_args
            )
        else:
            attr = compute_attr(
                inputs, baselines, explainer, additional_forward_args, self.args
            )
    
        results = []
        # get scores
        for area in self.args.areas:
            for metric_name in self.args.metrics:
                metric = expl_metric_map[metric_name]
                error_comp = metric(
                    self.model, inputs=inputs, 
                    attributions=attr, baselines=baselines, 
                    additional_forward_args=additional_forward_args,
                    topk=area, mask_largest=True
                )
                
                error_suff = metric(
                    self.model, inputs=inputs, 
                    attributions=attr, baselines=baselines, 
                    additional_forward_args=additional_forward_args,
                    topk=area, mask_largest=False
                )
        
                result_row = [batch_index, metric_name, area, error_comp, error_suff]
                results.append(result_row)
        return results
        
    def dump_results(self, results, columns, filename):
        results_df = pd.DataFrame(results, columns=columns)
        results_df = results_df.groupby(['metric', 'area'])[
            ['comp', 'suff']
        ].aggregate('mean').reset_index()
        
        filepath = os.path.join(self.result_folder, filename)
        results_df.round(6).to_csv(filepath, index=False)
        print(results_df)