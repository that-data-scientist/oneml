import pandas as pd

from src.utils.console import log_step


class Imputer():
    def __init__(self):
        mode_imputer = ModeImputer()
        mean_imputer = MeanImputer()
        self.imputing_strategies = [mode_imputer, mean_imputer]

    @log_step
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_value_pct = missing_value_percentage(df)

        if missing_value_pct.sum() == 0 or missing_value_pct.sum() < 0.1:
            return df.dropna()

        for k, v in missing_value_pct.items():
            for imputing_strategy in self.imputing_strategies:
                if imputing_strategy.check(df[k]) is True:
                    df[k] = imputing_strategy.impute(df[k])
        return df


class ImputingStrategy():
    def check(self, series: pd.Series) -> bool:
        pass

    def impute(self, series: pd.Series) -> pd.Series:
        pass


class ModeImputer(ImputingStrategy):
    def check(self, series: pd.Series) -> bool:
        if pd.api.types.is_string_dtype(series):
            return True

        return False

    def impute(self, series: pd.Series) -> pd.Series:
        return series.fillna(series.mode().iloc[0])


class MeanImputer(ImputingStrategy):
    def check(self, series: pd.Series) -> bool:
        if pd.api.types.is_numeric_dtype(series):
            return True

        return False

    def impute(self, series: pd.Series) -> pd.Series:
        return series.fillna(series.mean())


def missing_value_percentage(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum() / df.shape[0]
