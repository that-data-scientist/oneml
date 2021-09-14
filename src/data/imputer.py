import pandas as pd

from src.utils.console import log_step


class Imputer():
    def __init__(self):
        self.mode_imputer = ModeImputer()
        self.mean_imputer = MeanImputer()

    @log_step
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_value_pct = missing_value_percentage(df)

        if missing_value_pct.sum() == 0:
            return df

        if missing_value_pct.sum() < 0.1:
            return df.dropna()

        for k, v in missing_value_pct.items():
            if pd.api.types.is_string_dtype(df[k]):
                df[k] = self.mode_imputer.impute(df[k])
            elif pd.api.types.is_numeric_dtype(df[k]):
                df[k] = self.mean_imputer.impute(df[k])
        return df


class ImputingStrategy():
    def impute(self, series: pd.Series) -> pd.Series:
        pass


class ModeImputer(ImputingStrategy):
    def impute(self, series: pd.Series) -> pd.Series:
        return series.fillna(series.mode().iloc[0])


class MeanImputer(ImputingStrategy):
    def impute(self, series: pd.Series) -> pd.Series:
        return series.fillna(series.mean())


def missing_value_percentage(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum() / df.shape[0]
