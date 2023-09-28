import os, torch
from tqdm import tqdm 
import pandas as pd
from utils.explainer import *
from tint.metrics import mae, mse

expl_metric_map = {
    'mae': mae, 'mse': mse
}

class Exp_Interpret:
    def __init__(
        self, model, result_folder, device, args, 
        explainers, explainer_name_map, areas
    ) -> None:
        self.result_folder = result_folder
        self.device = device
        self.args = args
        self.explainers = explainers
        self.areas = areas
    
        assert not args.output_attention, 'Model need to output target only'
        model.eval()
        model.zero_grad()
        self.model = model
        
        self.explainers_map = dict()
        for name in explainers:
            self.explainers_map[name] = explainer_name_map[name](self.model)
    
    def interpret(self, dataloader, flag, tsr=False, baseline_mode='aug'):
        assert baseline_mode in ['zeros', 'aug'], \
            f'Supported baseline modes: zeros, aug. Found {baseline_mode}'
        
        results = []
        baseline_mode = "aug" # "zeros", "aug"
        result_columns = ['batch_index', 'explainer', 'metric', 'area', 'comp', 'suff']
        
        if tsr:
            output_filename = "batch_interpretations_tsr.csv"
        else:
            output_filename = "batch_interpretations.csv"
            
        output_file = open(os.path.join(self.result_folder, output_filename), 'w')
        output_file.write(','.join(result_columns))
        
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
            
            inputs = batch_x
            # baseline must be a scaler or tuple of tensors with same dimension as input
            if baseline_mode=='zeros': baselines = torch.zeros_like(inputs)
            else: baselines = torch.mean(inputs, axis=0).repeat(inputs.shape[0], 1, 1).float()
            
            additional_forward_args = (batch_x_mark, dec_inp, batch_y_mark)

            # get attributions
            for name in self.explainers:
                explainer = self.explainers_map[name]
                if tsr:
                    attr = compute_tsr_attr(
                        inputs, baselines, explainer, additional_forward_args, self.args
                    )
                else:
                    attr = compute_attr(
                        inputs, baselines, explainer, additional_forward_args, self.args
                    )
            
                # get scores
                for area in self.areas:
                    for metric_name, metric in expl_metric_map.items():
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
                
                        result_row = [batch_index, name, metric_name, area, error_comp, error_suff]
                        output_file.write("\n" + ','.join([str(r) for r in result_row]))
                        results.append(result_row)
                
                output_file.flush()
            # break
        output_file.close()
        if tsr:
            self.dump_results(results, result_columns, f'interpretations_tsr_{flag}.csv')
        else:
            self.dump_results(results, result_columns, f'interpretations_{flag}.csv')
        
    def dump_results(self, results, columns, filename):
        results_df = pd.DataFrame(results, columns=columns)
        results_df = results_df.groupby(['explainer', 'metric', 'area'])[
            ['comp', 'suff']
        ].aggregate('mean').reset_index()
        
        filepath = os.path.join(self.result_folder, filename)
        results_df.round(4).to_csv(filepath, index=False)
        print(results_df)