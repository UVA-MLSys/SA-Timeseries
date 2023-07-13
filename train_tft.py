import os, gc, torch

import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE, MultiLoss

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from configurations.config import *

parser = ArgumentParser(
    description='Train model', 
    formatter_class=ArgumentDefaultsHelpFormatter
)

parser.add_argument(
   '--experiment', metavar='-e', 
   default=ExperimentType.TRAFFIC, 
   choices=ExperimentType.values(),
   help='dataset name of the experiment'
)

parser.add_argument(
   '--disable-progress',
   action='store_false',
   help='disable the progress bar.'
)

arguments = parser.parse_args()
show_progress_bar = not arguments.disable_progress
config = ExperimentConfig(experiment=arguments.experiment)
formatter = config.data_formatter

# Check if running on cpu or gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

print(f'Using {device} backend.')

# Load dataset
df = formatter.load()
print(f'Total data shape {df.shape}')

from utils.metric import show_result
from utils.data import create_TimeSeriesDataSet
from utils.model import seed_torch
seed_torch(seed=config.seed)
train, validation, test = formatter.split(df)

parameters = config.model_parameters(ModelType.TFT)
batch_size = parameters['batch_size']
_, train_dataloader = create_TimeSeriesDataSet(
    train, formatter, batch_size, train=True
)
_, val_dataloader = create_TimeSeriesDataSet(validation, formatter, batch_size)
test_timeseries, test_dataloader = create_TimeSeriesDataSet(test, formatter, batch_size)

import tensorflow as tf
# click this and locate the lightning_logs folder path and select that folder. 
# this will load tensorbaord visualization
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0, 
    patience=parameters['early_stopping_patience']
    , verbose=True, mode="min"
)
best_checkpoint = pl.callbacks.ModelCheckpoint(
    dirpath=config.experiment_folder, monitor="val_loss", 
    filename="best-{epoch}"
)
latest_checkpoint = pl.callbacks.ModelCheckpoint(
    dirpath=config.experiment_folder, 
    every_n_epochs=1, filename="latest-{epoch}"
)

logger = TensorBoardLogger(config.experiment_folder)  # logging results to a tensorboard

# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api
trainer = pl.Trainer(
    max_epochs = parameters['epochs'],
    accelerator = 'auto',
    enable_model_summary=True,
    callbacks = [early_stop_callback, best_checkpoint, latest_checkpoint],
    logger = logger,
    enable_progress_bar = show_progress_bar,
    check_val_every_n_epoch = 1,
    gradient_clip_val=parameters['gradient_clip_val'],
    max_time=pd.to_timedelta(2, unit='minutes')
)

tft = TemporalFusionTransformer.from_dataset(
    test_timeseries,
    learning_rate= parameters['learning_rate'],
    hidden_size= parameters['hidden_layer_size'],
    attention_head_size=parameters['attention_head_size'],
    dropout=parameters['dropout_rate'],
    loss=MultiLoss([RMSE(reduction='mean') for _ in formatter.targets]), # RMSE(reduction='sqrt-mean')
    optimizer='adam',
    log_interval=1,
    # reduce_on_plateau_patience=2
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

from datetime import datetime

gc.collect()

start = datetime.now()
print(f'\n----Training started at {start}----\n')

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
end = datetime.now()
print(f'\n----Training ended at {end}, elapsed time {end-start}')
print(f'Best model by validation loss saved at {trainer.checkpoint_callback.best_model_path}')

from classes.PredictionProcessor import PredictionProcessor

processor = PredictionProcessor(
    formatter.time_index, formatter.group_id, 
    formatter.parameters['horizon'], formatter.targets, 
    formatter.parameters['window']
)

# %%
from classes.Plotter import *

plotter = PlotResults(
   config.experiment_folder, formatter.time_index, 
   formatter.targets, show=show_progress_bar
)

best_model_path = trainer.checkpoint_callback.best_model_path
print(f'Loading best model from {best_model_path}')

# tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# print('\n---Training prediction--\n')
# train_predictions, train_index = tft.predict(
#     train_dataloader, return_index=True, 
#     show_progress_bar=show_progress_bar
# )
# train_result_merged = processor.align_result_with_dataset(
#    train, train_predictions, train_index
# )

# show_result(train_result_merged, formatter.targets)
# plotter.summed_plot(train_result_merged, type='Train_error', plot_error=True)
# gc.collect()

print(f'\n---Validation results--\n')

validation_predictions, validation_index = tft.predict(
    val_dataloader, return_index=True, 
    show_progress_bar=show_progress_bar
)

validation_result_merged = processor.align_result_with_dataset(
   validation, validation_predictions, validation_index
)
show_result(validation_result_merged, formatter.targets)
plotter.summed_plot(validation_result_merged, type='Validation')
gc.collect()

print(f'\n---Test results--\n')

test_predictions, test_index = tft.predict(
    test_dataloader, return_index=True, 
    show_progress_bar=show_progress_bar
)

test_result_merged = processor.align_result_with_dataset(
    test, test_predictions, test_index
)
show_result(test_result_merged, formatter.targets)
plotter.summed_plot(test_result_merged, 'Test')
gc.collect()

# train_result_merged['split'] = 'train'
validation_result_merged['split'] = 'validation'
test_result_merged['split'] = 'test'
df = pd.concat([validation_result_merged, test_result_merged])
df.to_csv(os.path.join(plotter.figPath, 'predictions.csv'), index=False)

print(f'Ended at {datetime.now()}. Elapsed time {datetime.now() - start}')