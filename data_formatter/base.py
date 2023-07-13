import os
from abc import ABC, abstractmethod, abstractproperty
from typing import Union, List
from enum import Enum
import pandas as pd
from pandas import read_csv, to_datetime
import numpy as np
from utils.download import *
from tqdm import tqdm

DISABLE_PROGRESS = False

# Type defintions
class DataTypes(str, Enum):
  """Defines numerical types of each column."""
  INTEGER = 'int'
  FLOAT = 'float'
  CATEGORICAL = 'categorical'
  DATE = 'date'

  def __str__(self) -> str:
    return super().__str__()

class InputTypes(str, Enum):
  """
    Defines input types of each column.

    TARGET: Prediction target
    OBSERVED: Past dynamic inputs
    KNOWN: Known future values 
    STATIC: Static values
    ID: Single column used as an entity identifier
    TIME: Single column exclusively used as a time index
  """
  TARGET = 'target'
  OBSERVED = 'observed'
  KNOWN = 'known'
  STATIC = 'static'

  #TODO: add full support for multiple columns
  ID = 'id'  
  TIME = 'time' 

  def __str__(self) -> str:
    return super().__str__()

class BaseDataFormatter(ABC):
    """
    Abstract base class for all data formatters.

    User can implement the abstract methods below to 
    perform dataset-specific
    manipulations.
    """

    data_root = 'datasets'
    """Root directory of all datasets"""

    def __init__(self, data_folder:str = '') -> None:
        super().__init__()

        self.data_folder = os.path.join(self.data_root, data_folder)
        os.makedirs(self.data_folder,exist_ok=True)
    
    """Directory for input files, a subdir of the data_root"""

    def fix_column_types(
        self, df:pd.DataFrame
    ) -> pd.DataFrame:
        
        print('Feature column, Data type, Current type')
        for item in self.column_definition:
            key, data_type = item[0], item[1]
            print(key, data_type, df[key].dtype.name)
            
            if data_type == DataTypes.CATEGORICAL:
                df[key] = df[key].astype(str)
            elif data_type == DataTypes.DATE:
                df[key] = df[key].apply(to_datetime)
            elif data_type == DataTypes.INTEGER:
                df[key] = df[key].astype(int)
            elif data_type == DataTypes.FLOAT:
                df[key] = df[key].astype(float)
        
        print(df.dtypes)
        return df

    def load(self) -> pd.DataFrame:
        print(f'Loading {self.data_path}')

        if not os.path.exists(self.data_path):
            print(f'{self.data_path} not found.')
            df = self.download()

        df = read_csv(self.data_path)
        return self.fix_column_types(df)

    @property
    @abstractmethod
    def data_path(self):
        raise NotImplementedError()
    
    @abstractmethod
    def download(self, force=False) -> None:
        """Downloads the target file, preprocesses and dumps in the data folder. 
        Temporary files generated during the download are removed afterwards.

        Args:
            force (bool, optional): Force update current file. Defaults to False.

        Raises:
            NotImplementedError
        """        
        raise NotImplementedError()

    @property
    @abstractmethod
    def column_definition(self) -> list[tuple[Union[str, int], DataTypes, Union[InputTypes, list[InputTypes]]]]:
        """
        Defines feature, input type and data type of each column. 
        It is a list of tuples of the format (feature_name, data_type, input_type) 
        or (feature_name, data_type, list of input_types)
        """
        # https://www.geeksforgeeks.org/extract-multidict-values-to-a-list-in-python/
        raise NotImplementedError()
    
    @property
    def targets(self): return self.extract_columns(input_type=InputTypes.TARGET)

    #TODO: Add support for multiple group ids (e.g. in the prediction processor)
    @property
    def group_id(self):
        """
        Return the group id column where each id value represents one timeseries.
        """
        return self.extract_columns(input_type=InputTypes.ID)
    
    @property
    def time_index(self): return self.extract_columns(input_type=InputTypes.TIME)

    @property
    @abstractmethod
    def parameters(self):
        """Defines the fixed parameters used by the model for training.

        Returns:
        A dictionary of fixed parameters, e.g.:

        parameters = {
            'window': 1, # Length of input time sequence (past observations)
            'horizon': 1, # Length of output 
            'num_epochs': 10,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 1,
        }
        """
        raise NotImplementedError()
    
    @abstractmethod
    def split(self, data, train_start, val_start, test_start, test_end):
       """Performs the default train, validation and test splits.
        
        Args:
        df: Source data frame to split.
        valid_boundary: Starting year for validation data
        test_boundary: Starting year for test data

        Returns:
        Tuple of transformed (train, valid, test) data.
      """
       raise NotImplementedError()
    
    def extract_columns(
        self, data_type:Union[DataTypes, List[DataTypes]] = None, 
        input_type:Union[InputTypes, List[InputTypes]] = None
    )-> List[str]:
        """Extracts the names of columns that correspond to a define data_type.

        Args:
            definition: Column definition to use.
            data_type: DataType of columns to extract.
            input_type: InputType of columns to extract.

        Returns:
            Name or a list of names for columns with data and input type specified.
        """
        # print(f'\nExtracting data type {data_type}, input type {input_type}.')
        columns = []
        for item in self.column_definition:
            if data_type is not None:
                if isinstance(data_type, list):
                    found = [d for d in data_type if d in item]
                    if len(found) == 0: continue
                elif data_type not in item: continue

            if input_type is not None:
                if isinstance(input_type, list):
                    found = [d for d in input_type if d in item]
                    if len(found) == 0: continue
                    
                elif input_type not in item: continue

            columns.append(item[0])

        # print(f'Extracted columns {columns}.\n')
        # if len(columns)==1:
        #     return columns[0]
        
        return columns