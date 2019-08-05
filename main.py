from data_preprocessing.preprocessing import create_and_save_dataframes
from models.all_models import create_and_save_models


def main():
    """
    Main python file for creating all necessary data files
    """
    create_and_save_dataframes()
    create_and_save_models()


if __name__ == "__main__":
    main()
