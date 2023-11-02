import pandas as pd

def reduce_df(df:pd.DataFrame):
    # df[df['area']==0.05]
    return df.groupby('metric').aggregate('mean')[['comp', 'suff']].reset_index()

metric_map = {
    'electricity': ['mse','mae'],
    'traffic': ['mse', 'mae'],
    'mimic_iii': ['cross_entropy', 'auc', 'accuracy', ]
}

datasets = ['electricity', 'traffic', 'mimic_iii']
models = ['DLinear', 'MICN', 'LSTM', 'Crossformer']
attr_methods = [
    'feature_ablation', 'occlusion', 'augmented_occlusion', 
    'feature_permutation', 'winIT', 'tsr_orig'
]

short_form = {
    'feature_ablation': 'FA',
    'occlusion':'FO',
    'augmented_occlusion': 'AFO',
    'feature_permutation': 'FP',
    'winIT': 'WinIT',
    'tsr_orig':'TSR+IG'
}

for dataset in datasets:
    print(dataset)
    tsr_better_comp = 0
    tsr_better_suff = 0
    comp_count = suff_count = 0
    
    for attr_method in attr_methods:
        if attr_method in ['tsr_orig', 'winIT']: continue
        for model in models: # , 'Crossformer'
            df = pd.read_csv(f'results/{dataset}_{model}/{attr_method}.csv') 
            df = reduce_df(df)
            
            df_tsr = pd.read_csv(f'results/{dataset}_{model}/tsr_{attr_method}.csv')
            df_tsr = reduce_df(df_tsr)
            
            for metric in metric_map[dataset]:
                [comp, suff] = df[df['metric']==metric][['comp', 'suff']].values[0]
                # print(comp, suff)
                
                [tsr_comp, tsr_suff] = df_tsr[df_tsr['metric']==metric][['comp', 'suff']].values[0]
                # print(tsr_comp, tsr_suff)
                
                if metric in ['accuracy', 'auc']:
                    if comp > tsr_comp: tsr_better_comp +=1
                    if suff < tsr_suff: tsr_better_suff += 1
                else:
                    if comp < tsr_comp: tsr_better_comp +=1
                    if suff > tsr_suff: tsr_better_suff += 1
                comp_count +=1
                suff_count +=1
                break
                        
    print(f'TSR better for: comp {tsr_better_comp}, suff {tsr_better_suff}. Total cases: {suff_count}')
    print(f'TSR improve comprehensiveness on {100.0* tsr_better_comp/comp_count:0.4f}\%, \
        and sufficiency on {tsr_better_suff*100.0/suff_count:0.4f}\% cases.\n')

def print_row(item):
    print(f'& {item:0.4g} ', end='')
    
for dataset in datasets:
    print(f'printing latex table for {dataset} dataset.\n')
    print(f" & {' & '.join(models)} & {' & '.join(models)} \\\\ \\hline")
    
    tsr_better_comp = 0
    tsr_better_suff = 0
    comp_count = suff_count = 0
    
    for attr_method in attr_methods:
        print(f'{short_form[attr_method]} ', end='')
        for model in models:
            if attr_method == 'tsr_orig':
                df = pd.read_csv(f'results/{dataset}_{model}/tsr_integrated_gradients_orig.csv') 
            else:
                df = pd.read_csv(f'results/{dataset}_{model}/{attr_method}.csv') 
            df = reduce_df(df)
           
            for metric in metric_map[dataset]:
                [comp, _] = df[df['metric']==metric][['comp', 'suff']].values[0]
                if metric in ['auc', 'accuracy']:
                    comp = 1 - comp
                break
            print_row(comp)
            
        for model in models:
            if attr_method == 'tsr_orig':
                df = pd.read_csv(f'results/{dataset}_{model}/tsr_integrated_gradients_orig.csv') 
            else:
                df = pd.read_csv(f'results/{dataset}_{model}/{attr_method}.csv') 
            df = reduce_df(df)
            for metric in metric_map[dataset]:
                [_, suff] = df[df['metric']==metric][['comp', 'suff']].values[0]
                if metric in ['auc', 'accuracy']:
                    suff = 1 - suff
                break
            print_row(suff)
        print('\\\\')
    
    print('\\hline')  
    for attr_method in attr_methods:
        if attr_method in ['winIT', 'tsr_orig']: continue
        
        print(f'WTSR+{short_form[attr_method]} ', end='')
        for model in models:
            df_tsr = pd.read_csv(f'results/{dataset}_{model}/tsr_{attr_method}.csv')
            df_tsr = reduce_df(df_tsr)
            
            for metric in metric_map[dataset]:
                [tsr_comp, _] = df_tsr[df_tsr['metric']==metric][['comp', 'suff']].values[0]
                if metric in ['auc', 'accuracy']:
                    tsr_comp = 1 - tsr_comp
                break
            
            print_row(tsr_comp)
            
        for model in models:
            df_tsr = pd.read_csv(f'results/{dataset}_{model}/tsr_{attr_method}.csv')
            df_tsr = reduce_df(df_tsr)
            
            for metric in metric_map[dataset]:
                [_, tsr_suff] = df_tsr[df_tsr['metric']==metric][['comp', 'suff']].values[0]
                if metric in ['auc', 'accuracy']:
                    tsr_suff = 1 - tsr_suff
                break
            print_row(tsr_suff)
        
        print('\\\\')
    print('\\hline\n\n')
    # break
            
        
        