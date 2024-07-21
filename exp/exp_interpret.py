import os, torch, copy, gc
from tqdm import tqdm 
import pandas as pd
from utils.explainer import *
from attrs.tsr import TSR, TSR2
from attrs.winTSR import WinTSR
from attrs.winIT import WinIT
from attrs.wip import WinIT2, WinIT3
from tint.metrics import mae, mse, accuracy, cross_entropy, lipschitz_max, log_odds
from utils.auc import auc
from datetime import datetime
from captum.attr import (
    DeepLift,
    GradientShap,
    IntegratedGradients,
    Lime,
    FeaturePermutation
)
from pytorch_lightning import Trainer
from tint.attr import (
    AugmentedOcclusion,
    DynaMask,
    Fit,
    Occlusion, 
    FeatureAblation
)

expl_metric_map = {
    'mae': mae, 'mse': mse, 'accuracy': accuracy, 
    'cross_entropy':cross_entropy, 'lipschitz_max': lipschitz_max,
    'log_odds': log_odds, 'auc': auc, 
    # 'comprehensiveness': comprehensiveness, 'sufficiency': sufficiency    
}

explainer_name_map = {
    "deep_lift":DeepLift, # throws "One of the differentiated Tensors appears to not have been used in the graph"
    "gradient_shap":GradientShap,
    "integrated_gradients":IntegratedGradients,
    "lime":Lime, # very slow
    "occlusion":Occlusion,
    "augmented_occlusion":AugmentedOcclusion, # requires data when initializing
    # "dyna_mask":DynaMask, # Multiple inputs are not accepted for this method
    # "fit": Fit, # only supports classification
    "feature_ablation":FeatureAblation,
    "feature_permutation":FeaturePermutation,
    "winIT": WinIT,
    "tsr": TSR, "wtsr": WinTSR,
    'winIT2': WinIT2, 'winIT3': WinIT3, 'tsr2': TSR2
    # "ozyegen":FeatureAblation
}

class Exp_Interpret:
    def __init__(
        self, exp, dataloader
    ) -> None:
        assert not exp.args.output_attention, 'Model needs to output target only'
        self.exp = exp
        self.args = exp.args
        self.result_folder = exp.output_folder
        self.device = exp.device
        
        exp.model.eval()
        # exp.model.zero_grad()
        self.model = exp.model
        
        self.explainers_map = dict() 
        for name in exp.args.explainers:
            explainer = Exp_Interpret.initialize_explainer(
                name, exp.model, exp.args, exp.device, dataloader
            ) 
            self.explainers_map[name] = explainer
    
    @staticmethod
    def initialize_explainer(
        name, model, args, device, dataloader
    ):
        if name == 'deep_lift':
            # torch.backends.cudnn.enabled=False
            clone = copy.deepcopy(model)
            clone.train() # deep lift moedl needs to be in training mode
            explainer = explainer_name_map[name](clone)
            
        elif name == 'tsr':
            explainer = TSR(IntegratedGradients(model))
            
        elif name == 'wtsr':
            base_explainer = Exp_Interpret.initialize_explainer(
                'augmented_occlusion', model, args, device, dataloader
            ) 
            metric = 'js' if args.task_name == 'classification' else 'pd'
            explainer = WinTSR(base_explainer, metric)
            
        elif name in ['augmented_occlusion', 'winIT', 'fit', 'winIT2', 'winIT3']:
            add_x_mark = args.task_name != 'classification'
            all_inputs = get_total_data(dataloader, device, add_x_mark=add_x_mark)
            
            if name in ['winIT', 'winIT2', 'winIT3']:
                explainer = explainer_name_map[name](
                    model, all_inputs, args
                )
            elif name == 'fit':
                trainer = Trainer(
                    logger=False,
                    enable_progress_bar=False, max_epochs=5,
                    enable_model_summary=False
                )
                explainer = explainer_name_map[name](
                    model, features=all_inputs, trainer=trainer
                )
            else: explainer = explainer_name_map[name](
                model, data=all_inputs
            )
        elif name == 'tsr2':
            explainer = TSR2(model, args)
        else:
            explainer = explainer_name_map[name](model) 
        
        return explainer    
    
    def run_classifier(self, dataloader, name):
        results = [['batch_index', 'metric', 'tau', 'area', 'comp', 'suff']]
        attrs = []
        
        progress_bar = tqdm(
            enumerate(dataloader), total=len(dataloader), 
            disable=self.args.disable_progress
        )
        for batch_index, (batch_x, _, padding_mask) in progress_bar:
            batch_x = batch_x.float().to(self.device)
            padding_mask = padding_mask.float().to(self.device)
             
            inputs = batch_x
            # baseline must be a scaler or tuple of tensors with same dimension as input
            baselines = get_baseline(inputs, mode=self.args.baseline_mode)
            additional_forward_args = (padding_mask, None, None)

            # get attributions
            batch_results, batch_attr = self.evaluate(
                name, inputs, baselines, 
                additional_forward_args, batch_index
            )
            
            results.extend(batch_results)  
            attrs.append(batch_attr)
            
            if self.args.dry_run: break
        
        attrs = torch.vstack(attrs)
        return results, attrs
                
    def run_regressor(self, dataloader, name):
        results = [['batch_index', 'metric', 'tau', 'area', 'comp', 'suff']]
        attrs = []
        
        progress_bar = tqdm(
            enumerate(dataloader), total=len(dataloader), 
            disable=self.args.disable_progress
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
            baselines = get_baseline(inputs, mode=self.args.baseline_mode)
            additional_forward_args = (dec_inp, batch_y_mark)

            # get attributions
            batch_results, batch_attr = self.evaluate(
                name, inputs, baselines, 
                additional_forward_args, batch_index
            )
            results.extend(batch_results)
            attrs.append(batch_attr)
            if self.args.dry_run: break
        
        attrs = tuple(torch.vstack([a[i] for a in attrs]) for i in range(2))
        return results, attrs
    
    def interpret(self, dataloader):
        task = self.args.task_name
        
        for name in self.args.explainers:
            explainer_result_file = os.path.join(self.result_folder, f'{name}.csv')
            if (not self.args.overwrite) and os.path.exists(explainer_result_file): 
                print(f'{explainer_result_file} exists. Skipping ...')
                continue
            
            results = []
            start = datetime.now()
            print(f'Running {name} from {start}')
            
            if task == 'classification':
                results, attrs = self.run_classifier(dataloader, name)
            else:
                results, attrs = self.run_regressor(dataloader, name)
            
            end = datetime.now()
            print(f'Experiment ended at {end}. Total time taken {end - start}.')
            self.dump_results(results, f'{name}.csv')
                
            if self.args.dump_attrs:
                attr_output_file = f'{self.args.flag}_{name}.pt' 
                attr_output_path = os.path.join(self.result_folder, attr_output_file)
                
                if task == 'classification':
                    attr_numpy = [a.detach().cpu().numpy() for a in attrs]
                else:
                    attr_numpy = tuple([a.detach().cpu().numpy() for a in attrs])
                torch.save(attr_numpy, attr_output_path)
            
            gc.collect()
            print()
                
    def evaluate(
        self, name, inputs, baselines, 
        additional_forward_args, batch_index
    ):
        explainer = self.explainers_map[name]
        
        attr = compute_attr(
            name, inputs, baselines, explainer, 
            additional_forward_args, self.args
        )
    
        results = []
        
        pred_len = self.args.num_class if self.args.task_name == 'classification' else self.args.pred_len
        # get scores
        for tau in range(pred_len):
            if type(attr) == tuple:
                # batch x pred_len x seq_len x features
                attr_per_pred = tuple([
                    attr_[:, tau] for attr_ in attr
                ])
            else: attr_per_pred = attr[:, tau]
            
            for metric_name in self.args.metrics:    
                for area in self.args.areas:
                    metric = expl_metric_map[metric_name]
                    error_comp = metric(
                        self.model, inputs=inputs, 
                        attributions=attr_per_pred, baselines=baselines, 
                        additional_forward_args=additional_forward_args,
                        topk=area # , mask_largest=True # this is default
                    )
                    
                    if metric_name in ['comprehensiveness', 'sufficiency', 'log_odds']:
                        # these metrics doesn't have mask_largest parameter
                        error_suff = 0
                    else: error_suff = metric(
                        self.model, inputs=inputs,
                        attributions=attr_per_pred, baselines=baselines, 
                        additional_forward_args=additional_forward_args,
                        topk=area, mask_largest=False
                    )
            
                    result_row = [
                        batch_index, metric_name, tau, area, error_comp, error_suff
                    ]
                    results.append(result_row)
    
        return results, attr
        
    def dump_results(self, results, filename):
        results_df = pd.DataFrame(results[1:], columns=results[0])
        
        batch_filename = os.path.join(self.result_folder, f'batch_{filename}')
        results_df.round(6).to_csv(batch_filename, index=False)
        
        results_df = results_df.groupby(['metric', 'area'])[
            ['comp', 'suff']
        ].aggregate('mean').reset_index()
        
        filepath = os.path.join(self.result_folder, filename)
        results_df.round(6).to_csv(filepath, index=False)
        print(results_df)