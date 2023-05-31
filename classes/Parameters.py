import pandas as pd
from pandas import to_datetime
from typing import List
import json

class Preprocess:
    def __init__(
        self, data, remove_input_outliers, remove_target_outliers, 
        target_moving_average_by_day, scale_input, scale_target
    ):
        self.data = data 
        self.remove_input_outliers = remove_input_outliers
        self.remove_target_outliers = remove_target_outliers
        self.target_moving_average_by_day = target_moving_average_by_day 
        self.scale_input = scale_input 
        self.scale_target = scale_target
    
    def __str__(self) -> str:
        return json.dumps(self.data, indent=4)

class Split:
    def __init__(
        self, data, train_start, validation_start, test_start, test_end,
        first_date, last_date
    ):
        self.data = data
        
        self.location = [0.80, 0.10, 0.10]
        self.train_start = to_datetime(train_start)
        self.validation_start = to_datetime(validation_start )
        self.test_start = to_datetime(test_start )
        self.test_end = to_datetime(test_end)
        self.first_date = to_datetime(first_date)
        self.last_date = to_datetime(last_date)

        self.validation_end = self.test_start - pd.to_timedelta(1, unit='D')
        self.train_end = self.validation_start - pd.to_timedelta(1, unit='D')

    def __str__(self) -> str:
        return json.dumps(self.data, indent=4)

class DataParameters:
    def __init__(
        self, data, id, static_features_map, dynamic_features_map, known_futures, 
        target_map, time_idx, population, population_cut, split
    ):
        self.data = data
        self.id = id

        self.target_map = target_map
        self.targets = list(self.target_map.values())
        self.static_features_map = static_features_map
        self.static_features = self.get_static_real_features()

        self.dynamic_features_map = dynamic_features_map
        self.dynamic_features = self.get_dynamic_features() 
        
        self.time_varying_known_features = known_futures

        # uses past observations to predict future observations
        # the tensorflow TFT uses past observations as input by default
        # reference https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py#L735 
        self.time_varying_unknown_features = self.dynamic_features + self.targets

        self.time_idx = time_idx 
        self.population_filepath = population
        self.population_cut = population_cut 
        
        self.split = Split(split, **split)

    def get_static_real_features(self) -> List[str]:
        """Generates the list of static features

        Returns:
            list: feature names
        """

        features_map = self.static_features_map
        feature_list = []
        for value in features_map.values():
            if type(value)==list:
                feature_list.extend(value)
            else:
                feature_list.append(value)

        return feature_list

    def get_dynamic_features(self) -> List[str]:
        """Generates the list of dynamic features

        Returns:
            list: feature names
        """

        features_map = self.dynamic_features_map
        feature_list = []
        for value in features_map.values():
            if type(value)==list:
                feature_list.extend(value)
            else:
                feature_list.append(value)
        return feature_list

    def __str__(self) -> str:
        return json.dumps(self.data, indent=4)

class ModelParameters:
    def __init__(
        self, data:dict, hidden_layer_size, dropout_rate, input_sequence_length, target_sequence_length,
        epochs, attention_head_size, optimizer, learning_rate, clipnorm,
        early_stopping_patience, seed, batch_size
    ) :
        self.data = data

        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.input_sequence_length = input_sequence_length
        self.target_sequence_length = target_sequence_length

        self.epochs = epochs

        self.attention_head_size = attention_head_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.clipnorm = clipnorm
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.batch_size = batch_size

    def __str__(self) -> str:
        return json.dumps(self.data, indent=4)

class Parameters:
    def __init__(self, config, model_parameters, data, preprocess):
        self.config = config

        self.model_parameters = ModelParameters(model_parameters, **model_parameters)
        self.data = DataParameters(data, **data)
        self.preprocess = Preprocess(preprocess, **preprocess)

    def __str__(self) -> str:
        return json.dumps(self.config, indent=4)