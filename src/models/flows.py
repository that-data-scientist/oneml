from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error, mean_absolute_percentage_error

from src.models.metrics import RegressionMetrics, RegressionEvaluation
from src.models.regression_model import MeanModel
from src.utils.console import log_step


class Flow(metaclass=ABCMeta):

    @abstractmethod
    def run(self, df_train: pd.DataFrame, df_test: pd.DataFrame, target_variable: str) -> None:
        pass

    @abstractmethod
    def evaluate(self, observed: pd.Series, predicted: list) -> RegressionMetrics:
        pass


class RegressionFlow(Flow):

    @log_step
    def run(self, df_train: pd.DataFrame, df_test: pd.DataFrame, target_variable: str) -> RegressionEvaluation:
        mean_model = MeanModel()
        mean_model.train(df_train, target_variable)

        train_predictions = mean_model.predict(df_train)
        train_metrics = self.evaluate(df_train[target_variable], train_predictions)

        test_predictions = mean_model.predict(df_test)
        test_metrics = self.evaluate(df_test[target_variable], test_predictions)

        return RegressionEvaluation(mean_model.name, train_metrics, test_metrics)

    def evaluate(self, observed: pd.Series, predicted: list) -> RegressionMetrics:
        rmsle = np.sqrt(mean_squared_log_error(observed, predicted))
        mape = mean_absolute_percentage_error(observed, predicted)

        return RegressionMetrics(rmsle, mape)
