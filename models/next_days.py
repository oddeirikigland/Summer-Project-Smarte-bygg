import sys
import os
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../analysis")
from analysis.combined_dataset import get_holiday_data
from analysis.weather_data.weather_forecast import get_weather_forecast
from helpers import map_bool_to_int
from preprocessing.preprocessing import preprocess_data, preprocess_for_ml


def get_df_next_days():
    date_today = datetime.now()
    from_date = (date_today + timedelta(1)).date()
    to_date = (date_today + timedelta(7)).date()

    weather = get_weather_forecast()
    holiday = get_holiday_data(from_date, to_date)
    merged = pd.merge(
        weather, holiday, left_index=True, right_index=True, how="left"
    )
    merged["Canteen"] = -1
    map_bool_to_int(merged, "holiday")
    map_bool_to_int(merged, "vacation")
    map_bool_to_int(merged, "inneklemt")
    merged.index.name = "date"
    return merged


def save_dataframes():
    df = get_df_next_days()
    decision_tree_df = preprocess_data(df)
    ml_df = preprocess_for_ml(df)
    decision_tree_df.to_csv(
        index=True, path_or_buf="../data/decision_tree_df_next_days.csv"
    )
    ml_df.to_csv(index=True, path_or_buf="../data/ml_df_next_days.csv")


def main():
    save_dataframes()


if __name__ == "__main__":
    main()
