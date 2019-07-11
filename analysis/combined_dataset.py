import sys
import os
from weather_data.weather import get_split_weather_data
from holidays_data.holiday import create_dataframe
from parking_and_canteen import get_extended_canteen_data
import pandas as pd

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../helpers")
from helpers import map_bool_to_int
from constants import ROOT_DIR


def get_holiday_data(earliest_date, latest_date):
    holidays = []
    for i in range(int(earliest_date.year), int(latest_date.year) + 1):
        holidays.append(create_dataframe(i))
    holiday = pd.concat(holidays)
    holiday.index = pd.to_datetime(holiday.index).date
    return holiday


def get_weather_data(earliest_date, latest_date):
    weather = get_split_weather_data(
        str(earliest_date), str(latest_date), "{}/config.ini".format(ROOT_DIR)
    )
    weather.index = pd.to_datetime(weather.index).date
    weather.index.name = "date"
    return weather


def get_dataset():
    # Canteen data
    canteen = get_extended_canteen_data()
    canteen.index = pd.to_datetime(canteen.index).date
    canteen.index.name = "date"

    # Define date interval
    earliest_date = canteen.iloc[0].name
    latest_date = canteen.iloc[-1].name

    # Holiday data
    holiday = get_holiday_data(earliest_date, latest_date)

    # Weather data
    weather = get_weather_data(earliest_date, latest_date)

    # Merge weather and holiday through left outer join
    merged = pd.merge(
        weather, canteen, left_index=True, right_index=True, how="left"
    )
    merged = pd.merge(
        merged, holiday, left_index=True, right_index=True, how="left"
    )

    merged["Canteen"] = merged.apply(
        lambda row: row["Canteen"] if not row["holiday"] else 0.0, axis=1
    )

    map_bool_to_int(merged, "holiday")
    map_bool_to_int(merged, "vacation")
    map_bool_to_int(merged, "inneklemt")
    return merged.dropna()


def create_csv(filepath="{}/data/dataset.csv".format(ROOT_DIR)):
    result = get_dataset()
    result.to_csv(index=True, path_or_buf=filepath)


def open_csv(filepath="{}/data/dataset.csv".format(ROOT_DIR)):
    return pd.read_csv(filepath, index_col="date")


def main():
    create_csv()
    # result = get_dataset()
    # print(result)


if __name__ == "__main__":
    main()
