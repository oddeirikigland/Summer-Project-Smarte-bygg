import pandas as pd
from constants import ROOT_DIR
from helpers.helpers import all_paths_in_dir
import warnings

warnings.filterwarnings("ignore")


def get_sensor_files():
    sensor_files = all_paths_in_dir(
        "{}/data/raw_sensor_data/".format(ROOT_DIR), ".csv"
    )
    sensor_files.sort()
    hierarchy = pd.read_csv(sensor_files[0])
    locations = pd.read_csv(sensor_files[1])
    mapping = pd.read_csv(sensor_files[2])
    samples = pd.read_csv(sensor_files[3])
    sensors = pd.read_csv(sensor_files[4])

    return hierarchy, locations, mapping, samples, sensors


def main():
    print(get_sensor_files())


if __name__ == "__main__":
    main()
