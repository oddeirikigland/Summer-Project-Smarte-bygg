import pandas as pd
from analysis.weather_data.weather import get_split_weather_data
from analysis.holidays_data.holiday import create_dataframe
from analysis.parking_and_canteen import get_extended_canteen_data
from helpers.helpers import map_bool_to_int
from constants import ROOT_DIR


def get_holiday_data(earliest_date, latest_date):
    """
    Get holiday, vacation and 'inneklemte' data for several years
    :param earliest_date: int, first year of which to include holiday
    :param latest_date: int, last year of which to include holiday
    :return: a dataframe with holiday data
    """
    holidays = []
    for i in range(int(earliest_date.year), int(latest_date.year) + 1):
        holidays.append(create_dataframe(i))
    holiday = pd.concat(holidays)
    holiday.index = pd.to_datetime(holiday.index).date
    return holiday


def get_weather_data(earliest_date, latest_date):
    """
    Get weather data from a given period
    :param earliest_date: int, first year of which to include weather data
    :param latest_date: int, last year of which to include weather data
    :return: a dataframe with weather data
    """
    weather = get_split_weather_data(
        str(earliest_date), str(latest_date), "{}/config.ini".format(ROOT_DIR)
    )
    weather.index = pd.to_datetime(weather.index).date
    weather.index.name = "date"
    return weather


def get_dataset():
    """
    Merges all relevant data and stores it in dataset.csv so it should be easy to access later on.
        Relevant data includes canteen, weather and holiday data.
    :return: a dataframe with all relevant data
    """
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
    """
    Create csv file
    :param filepath: string of where to locate file
    """
    result = get_dataset()
    result.to_csv(index=True, path_or_buf=filepath)


def main():
    create_csv()
    # result = get_dataset()
    # print(result)


if __name__ == "__main__":
    main()
