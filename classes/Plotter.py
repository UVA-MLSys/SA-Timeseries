"""
Done following
https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting/models/base_model.html#BaseModel.plot_prediction
"""

import os, sys
import numpy as np
from pandas import DataFrame, to_timedelta
from typing import List, Dict
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
import matplotlib.pyplot as plt

sys.path.append('..')
from utils.metric import calculate_result
from classes.PredictionProcessor import *
from classes.PlotConfig import *

from matplotlib.ticker import StrMethodFormatter, MultipleLocator, ScalarFormatter

class PlotResults:
    def __init__(
        self, figPath:str, time_index, 
        targets:List[str], figsize=FIGSIZE, show=True
    ) -> None:
        self.figPath = figPath
        if not os.path.exists(figPath):
            print(f'Creating folder {figPath}')
            os.makedirs(figPath, exist_ok=True)

        self.figsize = figsize
        self.show = show
        self.targets = targets
        self.time_index = time_index
    
    def plot(
        self, df:DataFrame, target:str, title:str=None, scale=1, 
        base:int=None, figure_name:str=None, plot_error:bool=False,
        legend_loc='best'
    ):
        fig, ax = plt.subplots(figsize=self.figsize)
        if title is not None: plt.title(title)
        x_column = self.time_index

        plt.plot(df[x_column], df[target], color='blue', label='Ground Truth')
        plt.plot(df[x_column], df[f'Predicted_{target}'], color='green', label='Prediction')

        if plot_error:
            plt.plot(df[x_column], abs(df[target] - df[f'Predicted_{target}']), color='red', label='Error')
        _, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max*1.1)
        
        if base is None:
            x_first_tick = df[x_column].min()
            x_last_tick = df[x_column].max()
            x_major_ticks = 5
            ax.set_xticks(
                [x_first_tick + (x_last_tick - x_first_tick) * i / (x_major_ticks - 1) for i in range(x_major_ticks)]
            )
        else:
            ax.xaxis.set_major_locator(MultipleLocator(base=base))
        
        # plt.xticks(rotation = 15)
        plt.xlabel(x_column)
        ax.yaxis.set_major_formatter(get_formatter(scale))
        plt.ylabel(f'{target}')
            
        if plot_error:
            plt.legend(framealpha=0.3, edgecolor="black", ncol=3, loc=legend_loc)
        else:
            plt.legend(framealpha=0.3, edgecolor="black", ncol=2, loc=legend_loc)
            
        # fig.tight_layout() # might change y axis values

        if figure_name is not None:
            plt.savefig(os.path.join(self.figPath, figure_name), dpi=DPI)
        if self.show:
            plt.show()
        return fig

    def summed_plot(
        self, merged_df:DataFrame, type:str='', save:bool=True, 
        base:int=None, plot_error:bool=False, legend_loc='best'
    ):
        """
        Plots summation of prediction and observation from all counties

        Args:
            figure_name: must contain the figure type extension. No need to add target name as 
            this method will add the target name as prefix to the figure name.
        """
        summed_df = PredictionProcessor.makeSummed(
            merged_df, self.targets, self.time_index
        )
        figures = []
        for target in self.targets:
            predicted_column = f'Predicted_{target}'
            y_true, y_pred = merged_df[target].values, merged_df[predicted_column].values
            
            mae, rmse, rmsle, smape, r2 = calculate_result(y_true, y_pred)
            title = f'MAE {mae:0.3g}, RMSE {rmse:0.4g}, RMSLE {rmsle:0.3g}, SMAPE {smape:0.3g}, R2 {r2:0.3g}'
            
            if (summed_df[target].max() - summed_df[target].min()) >= 1e3:
                scale = 1e3
            else: scale = 1

            target_figure_name = None
            if save: target_figure_name = f'Summed_plot_{target}_{type}.jpg'

            fig = self.plot(
                summed_df, target, title, scale, base, target_figure_name, 
                plot_error, legend_loc
            )
            figures.append(fig)
        
        return figures