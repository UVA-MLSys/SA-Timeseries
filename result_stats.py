import pandas as pd
import numpy as np

def reduce_df(df:pd.DataFrame):
    # df[df['area']==0.05]
    return df.groupby('metric').aggregate('mean')[['comp', 'suff']].reset_index()

int_metric_map = {
    'electricity': ['mae'], #, 'mse'],
    'traffic': ['mae'], #, 'mse'],
    'mimic_iii': ['auc'] #, 'cross_entropy']
}

test_metric_map = {
    'electricity': ['mae', 'mse'],
    'traffic': ['mae', 'mse'],
    'mimic_iii': ['auc', 'accuracy']
}

datasets = ['electricity', 'traffic', 'mimic_iii']
models = ['DLinear', 'MICN', 'SegRNN', 'iTransformer']
attr_methods = [
    'feature_ablation', 'augmented_occlusion', 
    'feature_permutation',
    'integrated_gradients', 'gradient_shap', 'dyna_mask',
    'winIT', 'tsr' ,'wtsr'
]

short_form = {
    'feature_ablation': 'FA',
    'occlusion':'FO',
    'augmented_occlusion': 'AFO',
    'feature_permutation': 'FP',
    'winIT': 'WinIT',
    'tsr':'TSR',
    'wtsr': 'WinTSR',
    'gradient_shap': 'GS',
    'integrated_gradients': 'IG',
    'dyna_mask': 'DM'
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

def print_row(item, decimals=2):
    if type(item) == str:
        print(f'& {item} ', end='')    
    else: print(f'& {np.round(item, decimals):03} ', end='')
    
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
                
            print_row(scores / NUM_ITERATIONS, decimals=3)
        print('\\\\')
        
# this section finds the ranks for each method 
results = []
for dataset in datasets:
    for attr_method in attr_methods:
        for metric in int_metric_map[dataset]:
            for model in models:
                for itr_no in range(1, NUM_ITERATIONS+1):
                    df = pd.read_csv(f'results/{dataset}_{model}/{itr_no}/{attr_method}.csv')
                    df = reduce_df(df)
                    comp, suff= df[df['metric']==metric][['comp', 'suff']].values[0]
                    
                    results.append([
                        dataset, attr_method, metric, model, itr_no, comp, suff
                    ])

result_df = pd.DataFrame(results, columns=['dataset', 'attr_method', 'metric', 'model', 'itr_no', 'comp', 'suff'])
print(result_df.head(3))
result_df = result_df.groupby(['dataset', 'attr_method', 'metric', 'model'])[['comp', 'suff']].mean().reset_index()
result_df = result_df[result_df['metric'].isin(['mae', 'auc'])]

selected = result_df['metric'].isin(['auc', 'accuracy'])
result_df.loc[selected, ['comp', 'suff']] = 1 - result_df[selected][['comp', 'suff']]

result_df['comp_rank'] = result_df.groupby(['dataset', 'metric', 'model'])['comp'].rank(ascending=False)
result_df['suff_rank'] = result_df.groupby(['dataset', 'metric', 'model'])['suff'].rank(ascending=True)
result_df.groupby(['dataset', 'metric', 'attr_method'])[['comp_rank', 'suff_rank']].mean().reset_index()

df = pd.concat([
    result_df.drop(columns='suff_rank').rename(columns={'comp_rank': 'rank'}), 
    result_df.drop(columns='comp_rank').rename(columns={'suff_rank': 'rank'})
], axis=0)

ranks = df.groupby(['dataset', 'metric', 'attr_method'])['rank'].mean().round(1).reset_index(name='mean_rank')
ranks['rank'] = ranks.groupby(['dataset', 'metric'])['mean_rank'].rank()
    
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
                    
                        df = df[df['metric']==metric][['area', metric_type]]
                        dfs.append(df)
                
                    df = pd.concat(dfs, axis=0)
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    if metric in ['auc', 'accuracy']:
                        df[metric_type] = 1-df[metric_type]
                    
                    score = df[metric_type].mean()
                    print_row(score)
            
            mean_rank, rank = ranks[
                (ranks['dataset']==dataset) & (ranks['metric']==metric) & (ranks['attr_method']==attr_method)
            ][['mean_rank', 'rank']].values[0]
            
            print_row(f'{rank:.0f}({mean_rank})')
            print('\\\\')
        print('\\hline\n')

# print('\n\nResult after normalizing by baseline')
# results = {}
# for dataset in datasets:
#     # use the first or second on
#     for metric in int_metric_map[dataset]:
#         print(f'Dataset {dataset}, metric {metric}.\n')
#         print(f" & {' & '.join(models)} & {' & '.join(models)} \\\\ \\hline")
        
#         for attr_method in attr_methods:
#             print(f'{short_form[attr_method]} ', end='')
#             for metric_type in ['comp', 'suff']:
#                 for model in models:
#                     scores = []
#                     dfs = []
#                     for itr_no in range(1, NUM_ITERATIONS+1):
#                         df = pd.read_csv(f'results/{dataset}_{model}/{itr_no}/{attr_method}.csv') 
                    
#                         df = df[df['metric']==metric][['area', metric_type]]
#                         dfs.append(df)

#                     df = pd.concat(dfs, axis=0)
#                     df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    
#                     score = df[metric_type].mean()
#                     if metric in ['auc', 'accuracy']:
#                         score = 1-score
                    
#                     results[(dataset, metric, attr_method,metric_type, model)] = score
#                     baseline = results[(dataset, metric, 'feature_ablation', metric_type, model)]
#                     print_row(score/baseline)
#             print('\\\\')
#         print('\\hline\n')