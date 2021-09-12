import pandas as pd

from src.utils.console import log_step


class Imputer():
    def __init__(self):
        self.mode_imputer = ModeImputer()

    @log_step
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_value_pct = missing_value_percentage(df)

        if missing_value_pct.sum() == 0:
            return df

        if missing_value_pct.sum() < 0.1:
            return df.dropna()

        for k, v in missing_value_pct.items():
            df[k] = self.mode_imputer.impute(df[k])

        return df


class ImputingStrategy():
    def impute(self, series: pd.Series) -> pd.Series:
        pass


class ModeImputer(ImputingStrategy):
    def impute(self, series: pd.Series) -> pd.Series:
        return series.fillna(series.mode().iloc[0])


def missing_value_percentage(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum() / df.shape[0]
