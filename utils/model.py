import os
from typing import List
import pandas as pd
import pytorch_lightning as pl
import random

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    pl.seed_everything(seed)

def get_best_model_path(checkpoint_folder, prefix='best-epoch='):
        for item in os.listdir(checkpoint_folder):
            if item.startswith(prefix):
                print(f'Found saved model {item}.')
                return os.path.join(checkpoint_folder, item)

        raise FileNotFoundError(f"Couldn't find the best model in {checkpoint_folder}")

def upscale_prediction(targets:List[str], predictions, target_scaler, target_sequence_length:int):
    """
    if target was scaled, this inverse transforms the target. Also reduces the shape from
    (time, target_sequence_length, 1) to ((time, target_sequence_length)
    """
    if target_scaler is None:
        return [predictions[i].reshape((-1, target_sequence_length)) for i in range(len(targets))]

    df = pd.DataFrame({targets[i]: predictions[i].flatten() for i in range(len(targets))})
    df[targets] = target_scaler.inverse_transform(df[targets])

    return [df[target].values.reshape((-1, target_sequence_length)) for target in targets]