import os
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # choose input data based on Multivariate or Univariate setting
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # scale data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # add time encoding
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # select data split
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data


class MultiTimeSeries(Dataset):
    def __init__(
        self, root_path, flag='train', size=None,
        features='S', data_path='ETTh1.csv',
        target='OT', scale=True, timeenc=0, freq='d', 
        time_col='Date', id_col='FIPS', max_samples=-1
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        
        self.id_col = id_col
        self.time_col = time_col
        self.time_steps = self.seq_len + self.pred_len
        self.max_samples = max_samples
        self.scaler = StandardScaler()
        self.__read_data__()
        
    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))
        df_raw[self.time_col] = pd.to_datetime(df_raw[self.time_col])
        
        id_col, time_col, target, time_steps = self.id_col, self.time_col, self.target, self.time_steps
        df_raw.sort_values(by=time_col, inplace=True)
        input_cols = [
            col for col in df_raw.columns \
                if col not in [id_col, time_col, target]
        ]
            
        dates = df_raw[time_col].unique()
        num_total = len(dates)
        num_test = self.pred_len # int(len(dates) * 0.2)
        num_vali = self.pred_len # num_total - num_train - num_test
        num_train = num_total - num_test -  num_vali# int(len(dates) * 0.7)
        
        border1s = [0, num_train - self.seq_len, num_total - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, num_total]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        
        border1 = dates[border1s[self.set_type]]
        border2 = dates[border2s[self.set_type]-1]
        border1 = df_raw[time_col].values.searchsorted(border1, side='left')
        border2 = df_raw[time_col].values.searchsorted(border2, side='right')
        
        # get input features
        if self.features == 'M' or self.features == 'MS':
            selected_columns = input_cols+[target]
        elif self.features == 'S':
            selected_columns = [target]
        print('Selected columns ', selected_columns)
        self.selected_columns = selected_columns
            
        df_data = df_raw[border1:border2].copy().reset_index(drop=True)
        
        if self.scale:
            train_end = df_raw[time_col].values.searchsorted(
                dates[border2s[0]-1], side='right'
            )
            train_data = df_raw[0:train_end]
            self.scaler.fit(train_data[selected_columns])
            df_data.loc[:, selected_columns] = self.scaler.transform(df_data[selected_columns])
            
        # add time encoding
        data_stamp = self._add_time_features(df_data.loc[0, [self.time_col]])
        time_encoded_columns = data_stamp.shape[1]
        print('Number of time encoded columns :', time_encoded_columns)
        
        print('Getting valid sampling locations.')
        
        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in df_data.groupby(id_col):
            num_entries = len(df)
            if num_entries >= time_steps:
                valid_sampling_locations += [
                    (identifier, i)
                    for i in range(num_entries - time_steps + 1)
                ]
            split_data_map[identifier] = df

        max_samples = self.max_samples # -1 takes all samples
        
        if max_samples > 0 and len(valid_sampling_locations) > max_samples:
            print('Extracting {} samples...'.format(max_samples))
            ranges = [valid_sampling_locations[i] for i in np.random.choice(
                  len(valid_sampling_locations), max_samples, replace=False)]
        else:
            # print('Max samples={} exceeds # available segments={}'.format(
            #      max_samples, len(valid_sampling_locations)))
            ranges = valid_sampling_locations
            max_samples = len(valid_sampling_locations)
        
        self.data = np.zeros((max_samples, self.time_steps, len(selected_columns)))
        self.data_stamp = np.zeros((max_samples, self.time_steps, time_encoded_columns))
        for i, tup in enumerate(ranges):
            if ((i + 1) % 10000) == 0:
                print(i + 1, 'of', max_samples, 'samples done...')
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx:start_idx + time_steps]
            self.data[i, :, :] = sliced[selected_columns]
            self.data_stamp[i, :, :] = self._add_time_features(sliced[[self.time_col]])
        
    def __getitem__(self, index):
        s_end = self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[index][:s_end]
        seq_y = self.data[index][r_begin:r_end]
        
        seq_x_mark = self.data_stamp[index][:s_end]
        seq_y_mark = self.data_stamp[index][r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def _add_time_features(self, df):
        df_stamp = pd.DataFrame()
        df_stamp['date'] = pd.to_datetime(df[self.time_col])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        return data_stamp

    def __len__(self):
        return len(self.data) # - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data