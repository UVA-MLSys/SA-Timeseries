from classes.PlotConfig import *
import pandas as pd
import os
from matplotlib.ticker import MultipleLocator

model = 'tft_pytorch'
output_dir = os.path.join(model, 'comparisons')
metrics = ['MAE', 'RMSE', 'RMSLE', 'SMAPE', 'NNSE']

result = pd.read_csv(
    os.path.join(output_dir, 'performance_comparison.csv')
)

for metric in metrics:
    fig, ax = plt.subplots(figsize=(9,6))
    for method, selection_results in result.groupby('method'):
        ax.plot(
            selection_results['num_features'], selection_results[metric], label=method
        )

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax*1.005)

    ax.xaxis.set_major_locator(MultipleLocator(base=1))

    ax.set_ylabel(metric)
    ax.set_xlabel('Number of Features')
    ax.legend(ncol=2, edgecolor='black')
    
    fig.tight_layout()
    fig.savefig(f'{output_dir}/{metric}.jpg', dpi=200)