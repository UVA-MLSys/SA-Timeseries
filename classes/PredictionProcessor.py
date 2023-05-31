import pandas as pd
import numpy as np

class PredictionProcessor:
    """
    Converts the TFT output into plotable dataframe format
    """

    def __init__(
        self, time_idx, group_id, horizon, 
        targets, window
    ) -> None:
        #TODO: add support for multiple time index and group id
        self.time_idx = time_idx[0]
        self.group_id = group_id[0]

        self.horizon = horizon
        self.targets = targets
        self.window = window

    def convert_prediction_to_dict(
            self, predictions, index, target_time_step:int=None,
            remove_negative:bool = False
        ):
        time_index = index[self.time_idx].values
        ids = index[self.group_id].values

        if remove_negative:
            # set negative predictions to zero
            predictions[predictions<0] = 0

        predictions = predictions.numpy()
        results = {}

        # if you want result for only a specific horizon in the future
        if target_time_step is not None:
            assert 0 < target_time_step <= self.horizon,\
            f"Expects target time step within 1 and {self.horizon}, found {target_time_step}."

            # convert target day to index, as it starts from 0
            target_time_step -= 1
            for index in range(len(predictions)):
                # given time index is the time index of the first prediction
                # https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.base_model.BaseModel.html#pytorch_forecasting.models.base_model.BaseModel.predict
                current_time_index = time_index[index]
                current_id = ids[index]

                item = (current_id, current_time_index + target_time_step)
                predicted_value = predictions[index][target_time_step]
                results[item] = [predicted_value]

            return results

        # if you haven't specified a particular horizon, this returns all of them, 
        # so that you can take the average
        for index in range(len(predictions)):
            current_time_index = time_index[index]
            current_id = ids[index]

            for time_step in range(self.horizon):
                item = (current_id, current_time_index + time_step)
                predicted_value = predictions[index][time_step]

                if item in results:
                    results[item].append(predicted_value)
                else:
                    results[item] = [predicted_value]

        return results

    def convert_dict_to_dataframe(self, results:dict, feature_name:str):
        ids = []
        predictions = []
        time_index = []

        for key in results.keys():
            item = results[key]
            #TODO: more generalized
            ids.append(key[0])
            time_index.append(key[1])

            predictions.append(np.mean(item))
        
        result_df = pd.DataFrame({
            self.group_id: ids, self.time_idx: time_index,
            f'Predicted_{feature_name}': predictions  
        })
        return result_df

    def align_result_with_dataset(
        self, df, predictions, index, target_time_step:int = None,
        remove_negative:bool = False
    ):
        id_columns = list(index.columns)

        if type(predictions)==list:
            result_df = None
            for i, prediction in enumerate(predictions):
                prediction_df = self.convert_dict_to_dataframe(
                    self.convert_prediction_to_dict(
                        prediction, index, target_time_step, remove_negative
                    ),
                    self.targets[i]
                )
                if result_df is None:
                    result_df = prediction_df
                else:
                    result_df = result_df.merge(prediction_df, on=id_columns, how='inner')
        else:
            # when prediction is on a single target, e.g. cases
            result_df = self.convert_dict_to_dataframe(
                self.convert_prediction_to_dict(
                    predictions, index, target_time_step, remove_negative
                ),
                self.targets[0]
            )

        merged_data = result_df.merge(
            df[self.targets + id_columns], on=id_columns, how='inner'
        ).reset_index(drop=True)
        merged_data = merged_data.sort_values(by=id_columns).reset_index(drop=True)

        # round the values
        predicted_columns = [col for col in merged_data.columns if 'Predicted' in col]
        merged_data[predicted_columns] = merged_data[predicted_columns].round()
        
        return merged_data

    @staticmethod
    def makeSummed(df, targets, columns):
        predicted_columns = [col for col in df.columns if 'Predicted' in col]
        return df.groupby(columns)[predicted_columns + targets].aggregate('sum').reset_index()