import os, enum
from data_formatter.base import *
from data_formatter.electricity import ElectricityFormatter
from dataclasses import dataclass

class ExperimentType(str, enum.Enum):
    ELECTRICITY = 'electricity'

    @staticmethod
    def values():
        role_names = [member.value for _, member in ExperimentType.__members__.items()]
        return role_names

class ModelType(enum.auto):
    TFT = "tft"

class ExperimentConfig:
    seed = 7

    def __init__(
        self, experiment:ExperimentType=ExperimentType.ELECTRICITY, 
        root:str='outputs'
    ) -> None:
        self.experiment = experiment
        self.root = root
        
        self.experiment_folder = os.path.join(root, experiment)
        if not os.path.exists(self.experiment_folder):
            os.makedirs(self.experiment_folder, exist_ok=True)
        print(f'Model outputs will be saved at {self.experiment_folder}')
    
    @property
    def data_formatter(self):
        data_formatter_map = {
            ExperimentType.ELECTRICITY: ElectricityFormatter,
        }
        return data_formatter_map[self.experiment]()
    
    def model_parameters(self, model:ModelType=None):
        model_parameter_map = {
            ExperimentType.ELECTRICITY: ElectricModelParameters
        }
        parameter = None
        try:
            parameters = model_parameter_map[self.experiment]
            print(f"Experimental config found for {self.experiment}.")
            print(f"Fetching parameters from available models {list(parameters.keys())}.")
            parameter = parameters[model]
        except:
            raise ValueError("Experiment or model parameters not found !")
            
        return parameter

ElectricModelParameters = {
    ModelType.TFT: {
        "hidden_layer_size": 16,
        "dropout_rate": 0,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 100,
        "early_stopping_patience": 5,
        'attention_head_size': 4
    }
}