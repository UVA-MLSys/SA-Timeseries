import pandas as pd
import numpy as np

def reduce_df(df:pd.DataFrame):
    # df[df['area']==0.05]
    return df.groupby('metric').aggregate('mean')[['comp', 'suff']].reset_index()

metric_map = {
    'electricity': ['mae', 'mse'],
    'traffic': ['mae', 'mse'],
    'mimic_iii': ['auc', 'cross_entropy', 'accuracy', ]
}

datasets = ['electricity', 'traffic', 'mimic_iii']
models = ['DLinear', 'MICN', 'SegRNN', 'Crossformer']
attr_methods = [
    'feature_ablation', 'occlusion', 'augmented_occlusion', 
    'winIT', 'tsr' #,wtsr
]

short_form = {
    'feature_ablation': 'FA',
    'occlusion':'FO',
    'augmented_occlusion': 'AFO',
    'winIT': 'WinIT',
    'tsr':'TSR'
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
    print(f'& {np.round(item, 2):0.3g} ', end='')
    
for dataset in datasets:
    # use the first or second on
    for metric in metric_map[dataset]:
        print(f'Dataset {dataset}, metric {metric}.\n')
        print(f" & {' & '.join(models)} & {' & '.join(models)} \\\\ \\hline")
        
        tsr_better_comp = 0
        tsr_better_suff = 0
        comp_count = suff_count = 0
        
        for attr_method in attr_methods:
            print(f'{short_form[attr_method]} ', end='')
            for model in models:
                comps, suffs = [], []
                for itr_no in range(1, NUM_ITERATIONS+1):
                    df = pd.read_csv(f'results/{dataset}_{model}/{itr_no}/{attr_method}.csv') 
                    df = reduce_df(df)
                
                    comp, _ = df[df['metric']==metric][['comp', 'suff']].values[0]
                    if metric in ['auc', 'accuracy']:
                        comp = 1 - comp
                    comps.append(comp)

                    df = pd.read_csv(f'results/{dataset}_{model}/{itr_no}/{attr_method}.csv') 
                    df = reduce_df(df)
                    _, suff = df[df['metric']==metric][['comp', 'suff']].values[0]
                    if metric in ['auc', 'accuracy']:
                        suff = 1 - suff
                    suffs.append(suff)
                
                print_row(sum(comps)/NUM_ITERATIONS)
                print_row(sum(suffs)/NUM_ITERATIONS)
            print('\\\\')
        
            print('\\hline\n')  