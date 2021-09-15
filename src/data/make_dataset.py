import pandas as pd

from src.data.imputer import Imputer
from src.store.data_store import CsvStore
from src.utils.config import load_config


def main():
    store = CsvStore()
    config = load_config()
    imputer = Imputer()

    input_df = store.get_raw(project_name=config['project_name'], file_name=config['input_file_name'])

    processed_df = input_df \
        .pipe(imputer.handle_missing_values, config['missing_values_drop_threshold']) \
        .pipe(exclude_variables, config['exclude_variables'])

    store.put_processed(project_name=config['project_name'], file_name='dataset.csv', df=processed_df)


def exclude_variables(df: pd.DataFrame, exclude_variables: list) -> pd.DataFrame:
    return df.drop(columns=exclude_variables)


if __name__ == '__main__':
    main()
