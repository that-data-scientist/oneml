from src.data.clean_dataset import Imputer
from src.store.data_store import CsvStore
from src.utils.config import load_config


def main():
    store = CsvStore()
    config = load_config()
    imputer = Imputer()

    input_df = store.get_data(project_name=config['project_name'], file_path=config['input_file_name'])

    processed_df = input_df \
        .pipe(imputer.handle_missing_values)

    print(processed_df)


if __name__ == '__main__':
    main()
