import pandas as pd
from datetime import datetime, timedelta
from preprocessing.canteen_tail.canteen_tail import add_canteen_history
from preprocessing.decision_tree.decision_tree_preprocessing import (
    get_dataset_with_weekday,
)
from preprocessing.start_of_year.start_of_year import add_diff_from_start_year
from preprocessing.weather.categorize_weather import categorize_temperature
from analysis.combined_dataset import get_holiday_data
from analysis.weather_data.weather_forecast import get_weather_forecast
from analysis.combined_dataset import create_csv
from helpers.helpers import map_bool_to_int
from constants import ROOT_DIR


def from_intervall_to_col(df, new_col, old_col):
    df[new_col] = df.apply(
        lambda row: 1.0 if row[old_col] == new_col else 0.0, axis=1
    )
    return df


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


def preprocess_data(df):
    df = add_canteen_history(df)
    df = get_dataset_with_weekday(df)
    df = add_diff_from_start_year(df)
    df = categorize_temperature(df)
    df["precipitation"] = df["precipitation"].div(df["precipitation"].max())
    df["canteen_week_ago"] = df["canteen_week_ago"].div(
        df["canteen_week_ago"].max()
    )
    df["canteen_day_ago"] = df["canteen_day_ago"].div(
        df["canteen_day_ago"].max()
    )
    return df.fillna(0)


def preprocess_for_ml(df):
    from_intervall_to_col(df, "preferred_work_temp", "avg_temp")
    from_intervall_to_col(df, "stay_home_temp", "avg_temp")

    from_intervall_to_col(df, "Monday", "weekday")
    from_intervall_to_col(df, "Tuesday", "weekday")
    from_intervall_to_col(df, "Wednesday", "weekday")
    from_intervall_to_col(df, "Thursday", "weekday")
    from_intervall_to_col(df, "Friday", "weekday")
    from_intervall_to_col(df, "Saturday", "weekday")
    from_intervall_to_col(df, "Sunday", "weekday")

    df = df.drop(["avg_temp", "weekday"], axis=1)
    return df


def save_dataframes_next_days():
    df = get_df_next_days()
    decision_tree_df = preprocess_data(df)
    ml_df = preprocess_for_ml(decision_tree_df.copy())
    decision_tree_df.to_csv(
        index=True,
        path_or_buf="{}/data/decision_tree_df_next_days.csv".format(ROOT_DIR),
    )
    ml_df.to_csv(
        index=True, path_or_buf="{}/data/ml_df_next_days.csv".format(ROOT_DIR)
    )
    return decision_tree_df, ml_df


def create_and_save_dataframes():
    create_csv()
    dataframe = pd.read_csv(
        "{}/data/dataset.csv".format(ROOT_DIR), index_col="date"
    )

    df = dataframe.copy()
    decision_tree_df = preprocess_data(df)
    ml_df = preprocess_for_ml(decision_tree_df.copy())
    decision_tree_df.to_csv(
        index=True, path_or_buf="{}/data/decision_tree_df.csv".format(ROOT_DIR)
    )
    ml_df.to_csv(index=True, path_or_buf="{}/data/ml_df.csv".format(ROOT_DIR))
    save_dataframes_next_days()


def main():
    create_and_save_dataframes()


if __name__ == "__main__":
    main()
