import pandas as pd
import numpy as np

def reduce_df(df:pd.DataFrame):
    # df[df['area']==0.05]
    return df.groupby('metric').aggregate('mean')[['comp', 'suff']].reset_index()

int_metric_map = {
    'electricity': ['mae', 'mse'],
    'traffic': ['mae', 'mse'],
    'mimic_iii': ['auc', 'cross_entropy']
}

test_metric_map = {
    'electricity': ['mae', 'mse'],
    'traffic': ['mae', 'mse'],
    'mimic_iii': ['auc', 'accuracy']
}

datasets = ['electricity', 'traffic', 'mimic_iii']
models = ['DLinear', 'MICN', 'SegRNN']
attr_methods = [
    'feature_ablation', 'occlusion', 'augmented_occlusion', 
    'winIT', 'tsr' ,'wtsr'
]

short_form = {
    'feature_ablation': 'FA',
    'occlusion':'FO',
    'augmented_occlusion': 'AFO',
    'winIT': 'WinIT',
    'tsr':'TSR',
    'wtsr': 'WinTSR'
}
NUM_ITERATIONS = 3

# for dataset in datasets:
#     print(dataset)
#     wtsr_better_comp = 0
#     wtsr_better_suff = 0
#     comp_count = suff_count = 0
    
#     for itr_no in range(1,NUM_ITERATIONS+1):
#         for attr_method in attr_methods:
#             for model in models: # , 'Crossformer'
#                 df = pd.read_csv(f'results/{dataset}_{model}/{itr_no}/{attr_method}.csv') 
#                 df = reduce_df(df)
                
#                 df_wtsr = pd.read_csv(f'results/{dataset}_{model}/{itr_no}/wtsr.csv')
#                 df_wtsr = reduce_df(df_wtsr)
                
#                 for metric in metric_map[dataset]:
#                     [comp, suff] = df[df['metric']==metric][['comp', 'suff']].values[0]
#                     # print(comp, suff)
                    
#                     [wtsr_comp, wtsr_suff] = df_wtsr[df_wtsr['metric']==metric][['comp', 'suff']].values[0]
#                     # print(tsr_comp, tsr_suff)
                    
#                     if metric in ['accuracy', 'auc']:
#                         if comp > wtsr_comp: wtsr_better_comp +=1
#                         if suff < wtsr_suff: wtsr_better_suff += 1
#                     else:
#                         if comp < wtsr_comp: wtsr_better_comp +=1
#                         if suff > wtsr_suff: wtsr_better_suff += 1
#                     comp_count +=1
#                     suff_count +=1
#                     # break
                        
#     print(f'WinTSR better for: comp {wtsr_better_comp}, suff {wtsr_better_suff}. Total cases: {suff_count}')
#     print(f'WinTSR improve comprehensiveness on {100.0* wtsr_better_comp/comp_count:0.4f}\%, \
#         and sufficiency on {wtsr_better_suff*100.0/suff_count:0.4f}\% cases.\n')

def print_row(item):
    print(f'& {np.round(item, 3):0.3g} ', end='')
    
print(f"Dataset & Metric &" + " & ".join(models) + " \\\\ \\hline")
for dataset in datasets:
    for metric in test_metric_map[dataset]:
        print(dataset, ' & ', metric, end='')
        for model in models:
            
            scores = 0
            for itr_no in range(1, NUM_ITERATIONS+1):
                df = pd.read_csv(f'results/{dataset}_{model}/{itr_no}/test_metrics.csv')
                score = df[df['metric']==metric]['score'].values[0]
                scores += score
                
            print_row(scores / NUM_ITERATIONS)
        print('\\\\')
        
# results = []
# for dataset in datasets:
#     for attr_method in attr_methods:
#         for metric in int_metric_map[dataset]:
#             for model in models:
#                 for itr_no in range(1, NUM_ITERATIONS+1):
#                     df = pd.read_csv(f'results/{dataset}_{model}/{itr_no}/{attr_method}.csv')
#                     df = reduce_df(df)
#                     comp, suff= df[df['metric']==metric][['comp', 'suff']].values[0]
                    
#                     results.append([
#                         dataset, attr_method, metric, model, itr_no, comp, suff
#                     ])

# result_df = pd.DataFrame(results, columns=['dataset', 'attr_method', 'metric', 'model', 'itr_no', 'comp', 'suff'])
# result_df.to_csv('results.csv', index=False)
    
for dataset in datasets:
    # use the first or second on
    for metric in int_metric_map[dataset]:
        print(f'Dataset {dataset}, metric {metric}.\n')
        print(f" & {' & '.join(models)} & {' & '.join(models)} \\\\ \\hline")
        
        for attr_method in attr_methods:
            print(f'{short_form[attr_method]} ', end='')
            for metric_type in ['comp', 'suff']:
                for model in models:
                    scores = []
                    dfs = []
                    for itr_no in range(1, NUM_ITERATIONS+1):
                        df = pd.read_csv(f'results/{dataset}_{model}/{itr_no}/{attr_method}.csv') 
                        # df = reduce_df(df)
                    
                        df = df[df['metric']==metric][['area', metric_type]]
                        dfs.append(df)
                        # if metric in ['auc', 'accuracy']:
                        #     score = 1-score
                
                        # scores.append(score)
                    df = pd.concat(dfs, axis=0)
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    
                    score = df[metric_type].mean()
                    if metric in ['auc', 'accuracy']:
                        score = 1-score
                    
                    print_row(score)
            print('\\\\')
        print('\\hline\n')
