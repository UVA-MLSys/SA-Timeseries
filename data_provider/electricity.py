from data_provider.base import *
from pandas import DataFrame, to_datetime

class ElectricityFormatter(BaseDataFormatter):

    def __init__(self) -> None:
        super().__init__()
        self.data_folder = os.path.join(self.data_root, 'electricity') 

    @property
    def data_path(self):
        return os.path.join(self.data_folder, 'hourly_electricity.csv')

    @property
    def column_definition(self):
        return [
            ('id', DataTypes.INTEGER, InputTypes.ID),
            ('hours_from_start', DataTypes.INTEGER, InputTypes.TIME, InputTypes.KNOWN),
            ('power_usage', DataTypes.FLOAT, InputTypes.TARGET),
            ('hour', DataTypes.INTEGER, InputTypes.KNOWN),
            ('day_of_week', DataTypes.INTEGER, InputTypes.KNOWN),
            ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC),
        ]
    
    @property
    def parameters(self):
        return {
            "window": 7 * 24, # lag times hours
            "horizon": 24
        }
    
    def split(self, data:DataFrame, val_start=1315, test_start=1339):
        # this is done following Google's TFT paper
        # note that this is different from time index
        index = data['days_from_start']
        lags = 7

        train = data.loc[index < val_start].reset_index(drop=True)
        validation = data.loc[
            (index >= (val_start - lags)) & (index < test_start)
        ].reset_index(drop=True)
        test = data.loc[index >= (test_start-lags)].reset_index(drop=True)
        
        return train, validation, test
    
    def download(
            self, force=False, start='2014-01-01', end='2014-09-01'
    ) -> None:
        """Downloads electricity dataset from UCI repository."""

        if os.path.exists(self.data_path) and not force:
            return

        if force: print('Force updating current data.')

        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'

        csv_path = os.path.join(self.data_folder, 'LD2011_2014.txt')
        zip_path = csv_path + '.zip'

        download_and_unzip(url, zip_path, csv_path, self.data_folder)

        print('Aggregating to hourly data')

        df = pd.read_csv(csv_path, index_col=0, sep=';', decimal=',')
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Filter to match range used by other academic papers
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        df = df[(df.index >= start) & (df.index <=end)]
        print(f'Filtering out data outside {start} and {end}')

        # Used to determine the start and end dates of a series
        output = df.resample('1h').sum().fillna(0)

        earliest_time = output.index.min()

        df_list = []
        for label in tqdm(output, total=output.shape[1]):
            # print('Processing {}'.format(label))
            srs = output[label]

            start_date = min(srs.fillna(method='ffill').dropna().index)
            end_date = max(srs.fillna(method='bfill').dropna().index)

            active_range = (srs.index >= start_date) & (srs.index <= end_date)
            srs = srs[active_range].fillna(0)

            tmp = pd.DataFrame({'power_usage': srs})
            date = tmp.index
            tmp['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (
                date - earliest_time).days * 24
            tmp['days_from_start'] = (date - earliest_time).days
            tmp['date'] = date
            tmp['id'] = label
            tmp['hour'] = date.hour
            tmp['day'] = date.day
            tmp['day_of_week'] = date.dayofweek
            tmp['month'] = date.month

            df_list.append(tmp)

        output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)

        # Filter to match range used by other academic papers
        # output = output[(output['days_from_start'] >= 1096)
        #                 & (output['days_from_start'] < 1346)].copy()

        output.to_csv(self.data_path, index=False)
        cleanup(self.data_folder, self.data_path)

        print('Done.')
