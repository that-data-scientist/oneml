import unittest

import pandas as pd
from nose.tools import raises

from src.models.regression_model import MeanModel, ModelNotTrainedError


class TestRegressionModel(unittest.TestCase):
    def test_should_train_baseline_mean_model(self):
        input_df = self.get_example_dataframe()

        mean_model = MeanModel()
        mean_model.train(input_df, 'target')

        self.assertEqual(32, mean_model.mean_value)

    def test_should_predict_baseline_mean_value(self):
        input_df = self.get_example_dataframe()

        mean_model = MeanModel()
        mean_model.train(input_df, 'target')
        result_preds = mean_model.predict(input_df)

        self.assertEqual([32, 32, 32], result_preds)

    @raises(ModelNotTrainedError)
    def test_should_throw_not_trained_exception_if_predict_is_called_before_train(self):
        input_df = self.get_example_dataframe()

        mean_model = MeanModel()

        self.assertRaises(ModelNotTrainedError, mean_model.predict(input_df))

    def get_example_dataframe(self):
        input_df = pd.DataFrame(
            [
                {
                    'feature_1': 'some_value',
                    'target': 10
                },
                {
                    'feature_1': 'some_value',
                    'target': 50
                },
                {
                    'feature_1': 'some_value',
                    'target': 36
                }
            ]
        )

        return input_df
