from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from data_formatter.base import *
import pandas as pd

def create_TimeSeriesDataSet(
    data: pd.DataFrame, 
    formatter: BaseDataFormatter, 
    batch_size:int, train=False,
):
    data_timeseries = TimeSeriesDataSet(
        data,
        time_idx=formatter.time_index[0], # can't handle multiple time index
        target=formatter.targets,
        group_ids = formatter.group_id, 
        max_encoder_length=formatter.parameters['window'],
        max_prediction_length=formatter.parameters['horizon'],
        static_reals=formatter.extract_columns([DataTypes.INTEGER, DataTypes.FLOAT], InputTypes.STATIC),
        static_categoricals=formatter.extract_columns(DataTypes.CATEGORICAL, InputTypes.STATIC),
        time_varying_known_reals = formatter.extract_columns([DataTypes.INTEGER, DataTypes.FLOAT], InputTypes.KNOWN),
        time_varying_unknown_reals = formatter.extract_columns([DataTypes.INTEGER, DataTypes.FLOAT], InputTypes.OBSERVED),
        target_normalizer = MultiNormalizer(
            [GroupNormalizer(groups=formatter.group_id) for _ in range(len(formatter.targets))]
        )
    )

    if train:
        dataloader = data_timeseries.to_dataloader(
            train=True, batch_size=batch_size
        )
    else:
        dataloader = data_timeseries.to_dataloader(
            train=False, batch_size=batch_size*4
        )

    return data_timeseries, dataloader

def valid_date(date):
    try:
        pd.to_datetime(date)
        return True
    except:
        return False

def missing_percentage(df:pd.DataFrame):
    return df.isnull().mean().round(4).mul(100).sort_values(ascending=False)