import functools
import os
from abc import ABCMeta, abstractmethod

import pandas as pd

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DATA_DIR = os.path.join(PROJECT_DIR, "data")


class InvalidExtension(Exception):
    pass


def _check_filepath(ext):
    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            file_path = kwargs.get("file_path")
            if not file_path:
                file_path = args[1]

            if not file_path.endswith(ext):
                raise InvalidExtension(f"{file_path} has invalid extension, want {ext}")

            return f(*args, **kwargs)

        return _wrapper

    return _decorator


class Store(metaclass=ABCMeta):

    @abstractmethod
    def get_data(self, project_name: str, file_path: str, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def put_data(self, filepath: str, df: pd.DataFrame, **kwargs) -> None:
        pass

    def put_processed(self, project_name: str, file_path: str, df: pd.DataFrame, **kwargs):
        file_path = os.path.join(DATA_DIR, project_name, 'processed', file_path)
        self.put_data(file_path, df, **kwargs)


class CsvStore(Store):
    @_check_filepath(".csv")
    def get_data(self, project_name: str, file_path: str, **kwargs) -> pd.DataFrame:
        file_path = os.path.join(DATA_DIR, project_name, 'input', file_path)
        return pd.read_csv(file_path, **kwargs)

    @_check_filepath(".csv")
    def put_data(self, filepath: str, df: pd.DataFrame, **kwargs) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be of type pd.DataFrame, got {type(df)}")
        df.to_csv(filepath, index=False, **kwargs)
