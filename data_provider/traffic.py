from data_provider.base import *
from pandas import DataFrame

class TrafficFormatter(BaseDataFormatter):
    def __init__(self) -> None:
        super('traffic').__init__()

    @property
    def data_path(self):
        return os.path.join(self.data_folder, 'hourly_traffic.csv')

    @property
    def column_definition(self) -> dict:
        return [
            ('id', DataTypes.INTEGER, InputTypes.ID),
            ('hours_from_start', DataTypes.INTEGER, InputTypes.TIME, InputTypes.KNOWN),
            ('values', DataTypes.FLOAT, InputTypes.TARGET),
            ('time_on_day', DataTypes.INTEGER, InputTypes.KNOWN),
            ('day_of_week', DataTypes.INTEGER, InputTypes.KNOWN),
            ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC),
        ]
    
    @property
    def parameters(self) -> dict:
        return {
            "window": 7 * 24,
            "horizon": 24
        }
    
    def split(self, df, valid_boundary=151, test_boundary=166):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
        df: Source data frame to split.
        valid_boundary: Starting day for validation data
        test_boundary: Starting day for test data

        Returns:
        Tuple of transformed (train, valid, test) data.
        """

        print('Formatting train-valid-test splits.')

        index = df['sensor_day']
        train = df.loc[index < valid_boundary]
        validation = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test = df.loc[index >= test_boundary - 7]

        return train, validation, test
    
    def download(self, force=False) -> None:
        """Downloads traffic dataset from UCI repository."""
        if os.path.exists(self.data_path) and not force:
            return

        if force: print('Force updating current data.')

        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip'

        data_folder = self.data_folder
        csv_path = os.path.join(data_folder, 'PEMS_train')
        zip_path = os.path.join(data_folder, 'PEMS-SF.zip')

        download_and_unzip(url, zip_path, csv_path, data_folder)

        print('Aggregating to hourly data')

        def process_list(s, variable_type=int, delimiter=None):
            """Parses a line in the PEMS format to a list."""
            if delimiter is None:
                l = [
                    variable_type(i) for i in s.replace('[', '').replace(']', '').split()
                ]
            else:
                l = [
                    variable_type(i)
                    for i in s.replace('[', '').replace(']', '').split(delimiter)
                ]

            return l

        def read_single_list(filename):
            """Returns single list from a file in the PEMS-custom format."""
            with open(os.path.join(data_folder, filename), 'r') as dat:
                l = process_list(dat.readlines()[0])
            return l

        def read_matrix(filename):
            """Returns a matrix from a file in the PEMS-custom format."""
            array_list = []
            with open(os.path.join(data_folder, filename), 'r') as dat:
                lines = dat.readlines()
                for i, line in tqdm(enumerate(lines), disable=DISABLE_PROGRESS):
                    # if (i + 1) % 50 == 0:
                    #     print('Completed {} of {} rows for {}'.format(i + 1, len(lines),filename))

                    array = [
                        process_list(row_split, variable_type=float, delimiter=None)
                        for row_split in process_list(
                            line, variable_type=str, delimiter=';')
                    ]
                    array_list.append(array)

            return array_list

        shuffle_order = np.array(read_single_list('randperm')) - 1  # index from 0
        train_dayofweek = read_single_list('PEMS_trainlabels')
        train_tensor = read_matrix('PEMS_train')
        test_dayofweek = read_single_list('PEMS_testlabels')
        test_tensor = read_matrix('PEMS_test')

        # Inverse permutate shuffle order
        print('Shuffling')
        inverse_mapping = {
            new_location: previous_location
            for previous_location, new_location in enumerate(shuffle_order)
        }
        reverse_shuffle_order = np.array([
            inverse_mapping[new_location]
            for new_location, _ in enumerate(shuffle_order)
        ])

        # Group and reoder based on permuation matrix
        print('Reodering')
        day_of_week = np.array(train_dayofweek + test_dayofweek)
        combined_tensor = np.array(train_tensor + test_tensor)

        day_of_week = day_of_week[reverse_shuffle_order]
        combined_tensor = combined_tensor[reverse_shuffle_order]

        # Put everything back into a dataframe
        print('Parsing as dataframe')
        labels = ['traj_{}'.format(i) for i in read_single_list('stations_list')]

        hourly_list = []
        for day, day_matrix in enumerate(combined_tensor):

            # Hourly data
            hourly = pd.DataFrame(day_matrix.T, columns=labels)
            hourly['hour_on_day'] = [int(i / 6) for i in hourly.index
                                    ]  # sampled at 10 min intervals
            if hourly['hour_on_day'].max() > 23 or hourly['hour_on_day'].min() < 0:
                raise ValueError('Invalid hour! {}-{}'.format(
                    hourly['hour_on_day'].min(), hourly['hour_on_day'].max()))

            hourly = hourly.groupby('hour_on_day', as_index=True).mean()[labels]
            hourly['sensor_day'] = day
            hourly['time_on_day'] = hourly.index
            hourly['day_of_week'] = day_of_week[day]

            hourly_list.append(hourly)

        hourly_frame = pd.concat(hourly_list, axis=0, ignore_index=True, sort=False)

        # Flatten such that each entitiy uses one row in dataframe
        store_columns = [c for c in hourly_frame.columns if 'traj' in c]
        other_columns = [c for c in hourly_frame.columns if 'traj' not in c]
        flat_df = pd.DataFrame(columns=['values', 'prev_values', 'next_values'] +
                                other_columns + ['id'])

        def format_index_string(x):
            """Returns formatted string for key."""

            if x < 10:
                return '00' + str(x)
            elif x < 100:
                return '0' + str(x)
            elif x < 1000:
                return str(x)

            raise ValueError('Invalid value of x {}'.format(x))

        for store in tqdm(store_columns, disable=DISABLE_PROGRESS):
            # print('Processing {}'.format(store))

            sliced = hourly_frame[[store] + other_columns].copy()
            sliced.columns = ['values'] + other_columns
            sliced['id'] = int(store.replace('traj_', ''))

            # Sort by Sensor-date-time
            key = sliced['id'].apply(str) \
            + sliced['sensor_day'].apply(lambda x: '_' + format_index_string(x)) \
                + sliced['time_on_day'].apply(lambda x: '_' + format_index_string(x))
            sliced = sliced.set_index(key).sort_index()

            sliced['values'] = sliced['values'].fillna(method='ffill')
            sliced['prev_values'] = sliced['values'].shift(1)
            sliced['next_values'] = sliced['values'].shift(-1)

            flat_df = pd.concat([flat_df, sliced.dropna()], ignore_index=True, sort=False)

        # Filter to match range used by other academic papers
        index = flat_df['sensor_day']
        flat_df = flat_df[index < 173].copy()

        # Creating columns fo categorical inputs
        flat_df['categorical_id'] = flat_df['id'].copy()
        flat_df['hours_from_start'] = flat_df['time_on_day'] \
            + flat_df['sensor_day']*24

        flat_df.to_csv(self.data_path, index=False)
        cleanup(self.data_folder, self.data_path)
        print('Done.')
