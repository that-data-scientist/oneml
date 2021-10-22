import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.flows import RegressionFlow
from src.store.data_store import CsvStore
from src.utils.config import load_config


def main():
    store = CsvStore()
    config = load_config()

    processed_df = store.get_processed(project_name=config['project_name'], file_name='dataset.csv')
    df_train, df_test = train_test_split(processed_df, test_size=config["test_size"])

    target_variable = config["target_variable"]

    if pd.api.types.is_numeric_dtype(processed_df[target_variable]):
        regression_flow = RegressionFlow()
        regression_evaluation = regression_flow.run(df_train, df_test, target_variable)
        print(pd.DataFrame([regression_evaluation.to_dict()]))


if __name__ == '__main__':
    main()
