import unittest

import numpy as np
import pandas as pd
from numpy import NaN

from src.data.clean_dataset import missing_value_percentage, Imputer


class TestCleanDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.imputer = Imputer()

    def test_handle_missing_values_should_return_same_df_if_no_nan_fount(self):
        test_df = pd.DataFrame(
            [
                {
                    'key_1': 'value_1',
                    'key_2': 'value_2'
                },
                {
                    'key_1': 'value_1',
                    'key_2': 'value_2'
                }

            ]
        )

        result_df = self.imputer.handle_missing_values(test_df)

        pd.testing.assert_frame_equal(result_df, test_df)

    def test_handle_missing_values_should_return_same_df_if_no_nan_fount(self):
        test_df = pd.DataFrame(
            [
                {
                    'key_1': 'value_1',
                    'key_2': NaN
                },
                {
                    'key_1': NaN,
                    'key_2': 'value_2'
                }

            ]
        )

        result_df = self.imputer.handle_missing_values(test_df)

        self.assertEqual(0, result_df.isna().sum().sum())
        np.testing.assert_array_equal(np.array(['value_1', 'value_1']), result_df['key_1'].values)

    def test_missing_value_count(self):
        test_df = pd.DataFrame(
            [
                {
                    'key_1': 'value_1',
                    'key_2': NaN
                },
                {
                    'key_1': NaN,
                    'key_2': 'value_2'
                }

            ]
        )

        result = missing_value_percentage(test_df)

        pd.testing.assert_series_equal(
            pd.Series(data=pd.Series({'key_1': 0.5, 'key_2': 0.5})),
            result
        )
