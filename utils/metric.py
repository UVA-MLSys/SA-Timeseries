import torch
import pandas as pd
from pandas import DataFrame, to_datetime, to_timedelta
import numpy as np
from typing import List
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

def calculate_result(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    smape = symmetric_mean_absolute_percentage(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, rmsle, smape, r2

# https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.metrics.point.SMAPE.html?highlight=smape
def symmetric_mean_absolute_percentage(y_true, y_pred):
    value = 2*abs(y_true - y_pred) / (abs(y_true) + abs(y_pred))
    # for cases when both ground truth and predicted value are zero
    value = np.where(np.isnan(value), 0, value)
    
    return np.mean(value)

def show_result(df: pd.DataFrame, targets:List[str]):    
    for target in targets:
        predicted_column = f'Predicted_{target}'
        y_true, y_pred = df[target].values, df[predicted_column].values

        mae, rmse, rmsle, smape, r2 = calculate_result(y_true, y_pred)
        print(f'Target {target}, MAE {mae:.5g}, RMSE {rmse:.5g}, RMSLE {rmsle:0.5g}, SMAPE {smape:0.5g}. R2 {r2:0.5g}.')
    print()

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
