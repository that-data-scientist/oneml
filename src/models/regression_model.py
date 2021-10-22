from abc import abstractmethod, ABCMeta

import pandas as pd


class RegressionModel(metaclass=ABCMeta):

    @abstractmethod
    def train(self, df_train: pd.DataFrame, target_variable: str) -> None:
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> list:
        pass


class MeanModel(RegressionModel):
    def __init__(self):
        self.name = 'baseline_mean'
        self.mean_value = None

    def train(self, df: pd.DataFrame, target_variable: str) -> None:
        self.mean_value = df[target_variable].mean()

    def predict(self, df: pd.DataFrame) -> list:
        if self.mean_value is None:
            raise ModelNotTrainedError("Model has not been trained")

        predictions = [self.mean_value for _ in range(df.shape[0])]
        return predictions


class ModelNotTrainedError(RuntimeError):
    pass
