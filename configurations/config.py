import os, enum
from data_formatter.base import *
from data_formatter.electricity import ElectricityFormatter
from data_formatter.traffic import TrafficFormatter
from data_formatter.favorita import FavoritaFormatter
from data_formatter.volatility import VolatilityFormatter
from dataclasses import dataclass

class ExperimentType(str, enum.Enum):
    ELECTRICITY = 'electricity'
    TRAFFIC = 'traffic'
    FAVORITA = 'favorita'
    VOLATILITY = 'volatility'

    def __str__(self) -> str:
        return super().__str__()

    @staticmethod
    def values():
        role_names = [member.value for _, member in ExperimentType.__members__.items()]
        return role_names

class ModelType(enum.auto):
    TFT = "tft"

class ExperimentConfig:
    data_formatter_map = {
        ExperimentType.ELECTRICITY: ElectricityFormatter,
        ExperimentType.TRAFFIC: TrafficFormatter,
        ExperimentType.FAVORITA: FavoritaFormatter,
        # ExperimentType.VOLATILITY: VolatilityFormatter # volatility dataset unavailable 
    }

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
        return self.__class__.data_formatter_map[self.experiment]()
    
    def model_parameters(self, model:ModelType=None):
        model_parameter_map = {
            ExperimentType.ELECTRICITY: ElectricModelParameters,
            ExperimentType.TRAFFIC: TrafficModelParameters,
            ExperimentType.FAVORITA: FavoritaModelParameters,
            ExperimentType.VOLATILITY: VolatilityModelParameters
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
        'gradient_clip_val': 1,
        "early_stopping_patience": 5,
        'attention_head_size': 4
    }
}

VolatilityModelParameters = {
    ModelType.TFT: {
        'dropout_rate': 0.3,
        'hidden_layer_size': 160,
        'learning_rate': 0.01,
        'batch_size': 64,
        'gradient_clip_val': 1,
        "early_stopping_patience": 5,
        'attention_head_size': 1,
        'stack_size': 1,
        "epochs": 100,
    }
}

TrafficModelParameters = {
    ModelType.TFT: {
        'dropout_rate': 0.3,
        'hidden_layer_size': 320,
        'learning_rate': 0.001,
        'batch_size': 128,
        'gradient_clip_val': 100.,
        "early_stopping_patience": 5,
        'attention_head_size': 4,
        'stack_size': 1,
        "epochs": 100,
    }
}

FavoritaModelParameters = {
    ModelType.TFT: {
        'dropout_rate': 0.1,
        'hidden_layer_size': 240,
        'learning_rate': 0.001,
        'batch_size': 128,
        'gradient_clip_val': 100.,
        "early_stopping_patience": 5,
        'attention_head_size': 4,
        'stack_size': 1,
        "epochs": 100,
    }
}