from src.store.data_store import CsvStore
from src.utils.config import load_config


def main():
    store = CsvStore()
    config = load_config()

    input_df = store.get_data(project_name=config['project_name'], file_path=config['input_file_name'])
    print(input_df)


if __name__ == '__main__':
    main()
