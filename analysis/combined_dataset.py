from weather_data.weather import get_split_weather_data
from holidays.holiday import create_dataframe
from parking_and_canteen import get_extended_canteen_data
from datetime import date, datetime
import pandas as pd


def get_dataset():
    # Canteen data
    canteen = get_extended_canteen_data()
    canteen.index = pd.to_datetime(canteen.index).date
    canteen.index.name = "date"

    # Define date interval
    earliest_date = canteen.iloc[0].name
    date_today = date.today()

    # Holiday data
    holidays = []
    for i in range(int(earliest_date.year), int(date_today.year) + 1):
        holidays.append(create_dataframe(i))
    holiday = pd.concat(holidays)
    holiday.index = pd.to_datetime(holiday.index).date

    # Weather data
    weather = get_split_weather_data(
        str(earliest_date),
        str(date_today.strftime("%Y-%m-%d")),
        "../config.ini",
    )
    weather.index = pd.to_datetime(weather.index).date
    weather.index.name = "date"

    # Merge weather and holiday through left outer join
    merged = pd.merge(
        canteen, weather, left_index=True, right_index=True, how="left"
    )
    merged = pd.merge(
        merged, holiday, left_index=True, right_index=True, how="left"
    )

    merged["Canteen"] = merged.apply(
        lambda row: row["Canteen"] if not row["holiday"] else 0.0, axis=1
    )

    return merged.dropna()


def create_csv(filepath="../data/dataset.csv"):
    result = get_dataset()
    result.to_csv(index=True, path_or_buf=filepath)


def open_csv(filepath="../data/dataset.csv"):
    return pd.read_csv(filepath, index_col="date")


def main():
    create_csv()
    # result = get_dataset()
    # print(result)


if __name__ == "__main__":
    main()
