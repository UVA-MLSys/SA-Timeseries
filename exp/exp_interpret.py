import os, torch, copy, gc
from tqdm import tqdm 
import pandas as pd
import csv
from utils.explainer import *
from attrs.tsr import TSR
from attrs.winTSR import WinTSR
from attrs.winIT import WinIT
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
from exp.exp_basic import stringify_setting

expl_metric_map = {
    'mae': mae, 'mse': mse, 'accuracy': accuracy, 
    'cross_entropy':cross_entropy, 'lipschitz_max': lipschitz_max,
    'log_odds': log_odds, 'auc': auc, 
    # 'comprehensiveness': comprehensiveness, 'sufficiency': sufficiency    
}

explainer_name_map = {
    # "deep_lift":DeepLift, # throws RuntimeError: A Module Tanh() was detected that does not contain some of the input/output attributes that are required for DeepLift computations. This can occur, for example, if your module is being used more than once in the network.Please, ensure that module is being used only once in the network.
    "gradient_shap":GradientShap,
    "integrated_gradients":IntegratedGradients,
    "lime":Lime, # very slow
    "occlusion":Occlusion,
    "augmented_occlusion":AugmentedOcclusion, # requires data when initializing
    "dyna_mask":DynaMask, # Multiple inputs are not accepted for this method
    "fit": Fit, # only supports classification
    "feature_ablation":FeatureAblation,
    "feature_permutation":FeaturePermutation,
    "winIT": WinIT,
    "tsr": TSR, "wtsr": WinTSR
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
        # RuntimeError: cudnn RNN backward can only be called in training mode
        if name == 'deep_lift' or (('gradient' in name or name == 'dyna_mask') and 'RNN' in args.model):
            # torch.backends.cudnn.enabled=False
            clone = copy.deepcopy(model)
            clone.train() # deep lift moedl needs to be in training mode
            explainer = explainer_name_map[name](clone)
            
        elif name == 'tsr':
            explainer = TSR(model, args)
            
        elif name == 'wtsr':
            base_explainer = Exp_Interpret.initialize_explainer(
                'occlusion', model, args, device, dataloader
            ) 
            explainer = WinTSR(base_explainer)
            
        elif name in ['augmented_occlusion', 'winIT', 'fit', 'winIT2', 'winIT3']:
            add_x_mark = args.task_name != 'classification'
            all_inputs = get_total_data(dataloader, device, add_x_mark=add_x_mark)
            
            if name in ['winIT', 'winIT2', 'winIT3']:
                explainer = explainer_name_map[name](
                    model, all_inputs, args
                )
            elif name == 'fit':
                assert args.task_name == 'classification', 'fit only supports classification'
                # the parameters ensure Trainer doesn't flood the output with logs and create log folders
                trainer = Trainer(
                    logger=False, enable_checkpointing=False,
                    enable_progress_bar=False, max_epochs=5,
                    enable_model_summary=False,accelerator='auto'
                )
                # fit doesn't support multiple inputs
                explainer = explainer_name_map[name](
                    model, features=all_inputs, trainer=trainer
                )
            else: explainer = explainer_name_map[name](
                model, data=all_inputs
            )
        else:
            explainer = explainer_name_map[name](model) 
        
        return explainer    
    
    def run_classifier(self, dataloader, name):
        results = [['batch_index', 'metric', 'tau', 'area', 'comp', 'suff']]
        
        # this assumes the data is from same flag (train, val, test)
        batch_filename = os.path.join(self.result_folder, f'batch_{name}.csv')
        
        # no writing or resume needed for dry run
        if not self.args.dry_run:
            if os.path.exists(batch_filename) and (not self.args.overwrite): 
                results_df = pd.read_csv(batch_filename)
                # https://stackoverflow.com/questions/3348460/csv-file-written-with-python-has-blank-lines-between-each-row
                result_file = open(batch_filename, 'a', newline='')
                writer = csv.writer(result_file) 
                
                rows = results_df.values.tolist()
                results.extend(rows)
                if results_df.shape[0]>0: 
                    min_batch_index = results_df['batch_index'].max() + 1
            else:
                min_batch_index = 0
                # create and write header row if the file doesn't exists or it is to be overwritten
                result_file = open(batch_filename, 'w', newline='')
                writer = csv.writer(result_file) 
                writer.writerow(results[0])
            
        attrs = []
        if min_batch_index>0:
            print(f'Resuming from batch {min_batch_index}')
        
        progress_bar = tqdm(
            enumerate(dataloader), total=len(dataloader), 
            disable=self.args.disable_progress
        )
        
        for batch_index, (batch_x, _, padding_mask) in progress_bar:
            if batch_index < min_batch_index:
                continue
            
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
            gc.collect()
            
            if self.args.dry_run: break
            writer.writerows(batch_results)
            result_file.flush()
            
        attrs = torch.vstack(attrs)
        
        run_fraction = 1.0 * (batch_index + 1 - min_batch_index) / len(dataloader)
        return results, attrs, run_fraction
                
    def run_regressor(self, dataloader, name):
        results = [['batch_index', 'metric', 'tau', 'area', 'comp', 'suff']]
        
        # this assumes the data is from same flag (train, val, test)
        batch_filename = os.path.join(self.result_folder, f'batch_{name}.csv')
        
        min_batch_index = 0
        # no writing or resume needed for dry run
        if not self.args.dry_run:
            if os.path.exists(batch_filename) and not self.args.overwrite: 
                results_df = pd.read_csv(batch_filename)
                # https://stackoverflow.com/questions/3348460/csv-file-written-with-python-has-blank-lines-between-each-row
                result_file = open(batch_filename, 'a', newline='')
                writer = csv.writer(result_file) 
                
                rows = results_df.values.tolist()
                results.extend(rows)
                if results_df.shape[0]>0: 
                    min_batch_index = results_df['batch_index'].max() + 1
            else:
                # create and write header row if the file doesn't exists or it is to be overwritten
                result_file = open(batch_filename, 'w', newline='')
                writer = csv.writer(result_file) 
                writer.writerow(results[0])
        
        attrs = []
        if min_batch_index > 0:
            print(f'Resuming from batch {min_batch_index}')
            
        progress_bar = tqdm(
            enumerate(dataloader), total=len(dataloader), 
            disable=self.args.disable_progress
        )
            
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in progress_bar:
            if batch_index < min_batch_index:
                continue
            
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
            gc.collect()
            
            if self.args.dry_run: break
            # writing must appear after dry run break
            writer.writerows(batch_results)
            result_file.flush()
        
        attrs = tuple(torch.vstack([a[i] for a in attrs]) for i in range(2))
        run_fraction = 1.0* (batch_index + 1 - min_batch_index) / len(dataloader)
        return results, attrs, run_fraction
    
    def record_time_efficiency(self, start, end, name, run_fraction):
        if self.args.dry_run or (not self.args.overwrite and run_fraction == 0): return
        
        duration = end - start
        if run_fraction < 1:
            print(f'Method resumed at {1-run_fraction:.3f} fraction. Adjusting the time efficiency {duration} to {duration / run_fraction}.\n')
            duration = duration / run_fraction
                
        time_efficiency_file = os.path.join(
            self.args.result_path, f'time_efficiency.csv'
        )
        if not os.path.exists(time_efficiency_file):
            time_efficiency_file = open(time_efficiency_file, 'w', newline='')
            writer = csv.writer(time_efficiency_file) 
            writer.writerow(
                ['dataset', 'model', 'iteration', 
                 'name', 'timestamp', 'duration', 'settings']
            )
        else:
            time_efficiency_file = open(
                time_efficiency_file, 'a', newline=''
            )
            writer = csv.writer(time_efficiency_file)
        
        writer.writerow(
            [self.args.data_path.split('.')[0], self.args.model, 
             self.args.itr_no, name, end, duration, 
             stringify_setting(self.args, complete=True)
            ]
        )
        time_efficiency_file.flush()
        time_efficiency_file.close()
    
    def interpret(self, dataloader):
        task = self.args.task_name
        
        for name in self.args.explainers:
            explainer_result_file = os.path.join(self.result_folder, f'{name}.csv')
            explainer_batch_file = os.path.join(self.result_folder, f'batch_{name}.csv')
            if (not self.args.overwrite) and os.path.exists(explainer_result_file) and os.path.exists(explainer_batch_file):
                df = pd.read_csv(explainer_batch_file)
                if df.shape[0] > 0 and df['batch_index'].max() + 1 == len(dataloader):
                    print(f'{explainer_result_file} exists. Skipping ...')
                    continue
            
            start = datetime.now()
            print(f'\nRunning {name} from {start}')
            
            if task == 'classification':
                results, attrs, run_fraction = self.run_classifier(dataloader, name)
            else:
                results, attrs, run_fraction = self.run_regressor(dataloader, name)
            
            # this might not reflect the correct time if the results are resumed from a checkpoint
            end = datetime.now()
            print(f'Experiment ended at {end}. Total time taken {end - start}.')
            self.record_time_efficiency(start, end, name, run_fraction)
            
            if not self.args.dry_run:
                self.dump_results(results, f'{name}.csv')
            else:
                results_df = pd.DataFrame(results[1:], columns=results[0])
                results_df = results_df.groupby(['metric', 'area'])[
                    ['comp', 'suff']
                ].aggregate('mean').reset_index()
                print('Dry run results')
                print(results_df)
                
            # don't dump attr if dry run
            if self.args.dump_attrs and not self.args.dry_run:
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
                        topk=area, mask_largest=True # this is default
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
                        batch_index, metric_name, tau, area, 
                        np.round(error_comp, 6), np.round(error_suff, 6)
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