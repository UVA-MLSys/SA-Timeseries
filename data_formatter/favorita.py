from data_formatter.base import *
from pandas import DataFrame
import glob, gc

class FavoritaFormatter(BaseDataFormatter):
    def __init__(self) -> None:
        super().__init__('favorita')

    @property
    def data_path(self):
        return os.path.join(self.data_folder, 'favorita_consolidated.csv')

    @property
    def column_definition(self) -> dict:
        return [
            ('traj_id', DataTypes.INTEGER, InputTypes.ID),
            ('date', DataTypes.DATE, InputTypes.TIME),
            ('log_sales', DataTypes.FLOAT, InputTypes.TARGET),
            ('onpromotion', DataTypes.CATEGORICAL, InputTypes.KNOWN),
            ('transactions', DataTypes.INTEGER, InputTypes.OBSERVED),
            ('oil', DataTypes.INTEGER, InputTypes.OBSERVED),
            ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN),
            ('day_of_month', DataTypes.INTEGER, InputTypes.KNOWN),
            ('month', DataTypes.INTEGER, InputTypes.KNOWN),
            ('national_hol', DataTypes.CATEGORICAL, InputTypes.KNOWN),
            ('regional_hol', DataTypes.CATEGORICAL, InputTypes.KNOWN),
            ('local_hol', DataTypes.CATEGORICAL, InputTypes.KNOWN),
            ('open', DataTypes.INTEGER, InputTypes.KNOWN),
            ('item_nbr', DataTypes.CATEGORICAL, InputTypes.STATIC),
            ('store_nbr', DataTypes.CATEGORICAL, InputTypes.STATIC),
            ('city', DataTypes.CATEGORICAL, InputTypes.STATIC),
            ('state', DataTypes.CATEGORICAL, InputTypes.STATIC),
            ('type', DataTypes.CATEGORICAL, InputTypes.STATIC),
            ('cluster', DataTypes.CATEGORICAL, InputTypes.STATIC),
            ('family', DataTypes.CATEGORICAL, InputTypes.STATIC),
            ('class', DataTypes.CATEGORICAL, InputTypes.STATIC),
            ('perishable', DataTypes.CATEGORICAL, InputTypes.STATIC)
        ]
    
    @property
    def parameters(self) -> dict:
        return {
            "window": 90,
            "horizon": 30
        }
    
    def split(self, data, train_start, val_start, test_start, test_end):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
        df: Source data frame to split.
        valid_boundary: Starting year for validation data
        test_boundary: Starting year for test data

        Returns:
        Tuple of transformed (train, valid, test) data.
        """

        print('Formatting train-valid-test splits.')

        if valid_boundary is None:
            valid_boundary = pd.datetime(2015, 12, 1)

        fixed_params = self.get_fixed_params()
        time_steps = fixed_params['total_time_steps']
        lookback = fixed_params['num_encoder_steps']
        forecast_horizon = time_steps - lookback

        data['date'] = pd.to_datetime(data['date'])
        df_lists = {'train': [], 'valid': [], 'test': []}
        for _, sliced in data.groupby('traj_id'):
            index = sliced['date']
            train = sliced.loc[index < valid_boundary]
            train_len = len(train)
            valid_len = train_len + forecast_horizon
            valid = sliced.iloc[train_len - lookback:valid_len, :]
            test = sliced.iloc[valid_len - lookback:valid_len + forecast_horizon, :]

        sliced_map = {'train': train, 'valid': valid, 'test': test}

        for k in sliced_map:
            item = sliced_map[k]

            if len(item) >= time_steps:
                df_lists[k].append(item)

        dfs = {k: pd.concat(df_lists[k], axis=0) for k in df_lists}

        train = dfs['train']
        # self.set_scalers(train, set_real=True)

        # Use all data for label encoding  to handle labels not present in training.
        # self.set_scalers(data, set_real=False)

        # Filter out identifiers not present in training (i.e. cold-started items).
        def filter_ids(frame):
            identifiers = set(self.identifiers)
            index = frame['traj_id']
            return frame.loc[index.apply(lambda x: x in identifiers)]

        valid = filter_ids(dfs['valid'])
        test = filter_ids(dfs['test'])

        return train, valid, train


    def download(self, force=False) -> None:
        """Processes Favorita dataset.

        Makes use of the raw files should be manually downloaded from Kaggle @
            https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data

        Args:
            config: Default experiment config for Favorita
        """

        url = 'https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data'

        data_folder = self.data_folder

        # Save manual download to root folder to avoid deleting when re-processing.
        zip_file = os.path.join(data_folder, '..',
                                'favorita-grocery-sales-forecasting.zip')

        if not os.path.exists(zip_file):
            raise ValueError(
                f'Favorita zip file not found in {zip_file}!\
                Please manually download data from {url}.')

        # Unpack main zip file
        outputs_file = os.path.join(data_folder, 'train.csv.7z')
        unzip(zip_file, outputs_file, data_folder)

        # Unpack individually zipped files
        for file in glob.glob(os.path.join(data_folder, '*.7z')):

            csv_file = file.replace('.7z', '')

            unzip(file, csv_file, data_folder)

        print('Unzipping complete, commencing data processing...')

        # Extract only a subset of data to save/process for efficiency
        start_date = pd.datetime(2015, 1, 1)
        end_date = pd.datetime(2016, 6, 1)

        print('Regenerating data...')

        # load temporal data
        temporal = pd.read_csv(os.path.join(data_folder, 'train.csv'), index_col=0)

        store_info = pd.read_csv(os.path.join(data_folder, 'stores.csv'), index_col=0)
        oil = pd.read_csv(
            os.path.join(data_folder, 'oil.csv'), index_col=0).iloc[:, 0]
        holidays = pd.read_csv(os.path.join(data_folder, 'holidays_events.csv'))
        items = pd.read_csv(os.path.join(data_folder, 'items.csv'), index_col=0)
        transactions = pd.read_csv(os.path.join(data_folder, 'transactions.csv'))

        # Take first 6 months of data
        temporal['date'] = pd.to_datetime(temporal['date'])

        # Filter dates to reduce storage space requirements
        if start_date is not None:
            temporal = temporal[(temporal['date'] >= start_date)]
        if end_date is not None:
            temporal = temporal[(temporal['date'] < end_date)]

        dates = temporal['date'].unique()

        # Add trajectory identifier
        temporal['traj_id'] = temporal['store_nbr'].apply(
            str) + '_' + temporal['item_nbr'].apply(str)
        temporal['unique_id'] = temporal['traj_id'] + '_' + temporal['date'].apply(
            str)

        # Remove all IDs with negative returns
        print('Removing returns data')
        min_returns = temporal['unit_sales'].groupby(temporal['traj_id']).min()
        valid_ids = set(min_returns[min_returns >= 0].index)
        selector = temporal['traj_id'].apply(lambda traj_id: traj_id in valid_ids)
        new_temporal = temporal[selector].copy()
        del temporal
        gc.collect()
        temporal = new_temporal
        temporal['open'] = 1

        # Resampling
        print('Resampling to regular grid')
        resampled_dfs = []
        for traj_id, raw_sub_df in temporal.groupby('traj_id'):
            print('Resampling', traj_id)
            sub_df = raw_sub_df.set_index('date', drop=True).copy()
            sub_df = sub_df.resample('1d').last()
            sub_df['date'] = sub_df.index
            sub_df[['store_nbr', 'item_nbr', 'onpromotion']] \
                = sub_df[['store_nbr', 'item_nbr', 'onpromotion']].fillna(method='ffill')
            sub_df['open'] = sub_df['open'].fillna(
                0)  # flag where sales data is unknown
            sub_df['log_sales'] = np.log(sub_df['unit_sales'])

            resampled_dfs.append(sub_df.reset_index(drop=True))

        new_temporal = pd.concat(resampled_dfs, axis=0)
        del temporal
        gc.collect()
        temporal = new_temporal

        print('Adding oil')
        oil.name = 'oil'
        oil.index = pd.to_datetime(oil.index)
        temporal = temporal.join(
            oil.loc[dates].fillna(method='ffill'), on='date', how='left')
        temporal['oil'] = temporal['oil'].fillna(-1)

        print('Adding store info')
        temporal = temporal.join(store_info, on='store_nbr', how='left')

        print('Adding item info')
        temporal = temporal.join(items, on='item_nbr', how='left')

        transactions['date'] = pd.to_datetime(transactions['date'])
        temporal = temporal.merge(
            transactions,
            left_on=['date', 'store_nbr'],
            right_on=['date', 'store_nbr'],
            how='left')
        temporal['transactions'] = temporal['transactions'].fillna(-1)

        # Additional date info
        temporal['day_of_week'] = pd.to_datetime(temporal['date'].values).dayofweek
        temporal['day_of_month'] = pd.to_datetime(temporal['date'].values).day
        temporal['month'] = pd.to_datetime(temporal['date'].values).month

        # Add holiday info
        print('Adding holidays')
        holiday_subset = holidays[holidays['transferred'].apply(
            lambda x: not x)].copy()
        holiday_subset.columns = [
            s if s != 'type' else 'holiday_type' for s in holiday_subset.columns
        ]
        holiday_subset['date'] = pd.to_datetime(holiday_subset['date'])
        local_holidays = holiday_subset[holiday_subset['locale'] == 'Local']
        regional_holidays = holiday_subset[holiday_subset['locale'] == 'Regional']
        national_holidays = holiday_subset[holiday_subset['locale'] == 'National']

        temporal['national_hol'] = temporal.merge(
            national_holidays, left_on=['date'], right_on=['date'],
            how='left')['description'].fillna('')
        temporal['regional_hol'] = temporal.merge(
            regional_holidays,
            left_on=['state', 'date'],
            right_on=['locale_name', 'date'],
            how='left')['description'].fillna('')
        temporal['local_hol'] = temporal.merge(
            local_holidays,
            left_on=['city', 'date'],
            right_on=['locale_name', 'date'],
            how='left')['description'].fillna('')

        temporal.sort_values('unique_id', inplace=True)

        print('Saving processed file to {}'.format(self.data_path))
        temporal.round(6).to_csv(self.data_path, index=False)
        print('Done.')