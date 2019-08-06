import pandas as pd
from datetime import datetime, timedelta
from data_preprocessing.canteen_tail.canteen_tail import add_canteen_history
from data_preprocessing.decision_tree.decision_tree_preprocessing import (
    get_dataset_with_weekday,
)
from data_preprocessing.start_of_year.start_of_year import (
    add_diff_from_start_year,
)
from analysis.combined_dataset import get_holiday_data
from analysis.combined_dataset import create_csv
from helpers.helpers import map_bool_to_int
from constants import ROOT_DIR, DAYS_TO_TEST


def from_intervall_to_col(df, new_col, old_col):
    df[new_col] = df.apply(
        lambda row: 1.0 if row[old_col] == new_col else 0.0, axis=1
    )
    return df


def get_df_next_days():
    date_today = datetime.now()
    from_date = date_today.date()
    to_date = (date_today + timedelta(DAYS_TO_TEST)).date()

    holiday = get_holiday_data(from_date, to_date)

    mask = (holiday.index >= from_date) & (holiday.index <= to_date)

    holiday = holiday.loc[mask]
    holiday["Canteen"] = -1
    map_bool_to_int(holiday, "holiday")
    map_bool_to_int(holiday, "vacation")
    map_bool_to_int(holiday, "inneklemt")
    holiday.index.name = "date"
    return holiday


def preprocess_data(df):
    df = add_canteen_history(df)
    df = get_dataset_with_weekday(df)
    df = add_diff_from_start_year(df)
    df["canteen_week_ago"] = df["canteen_week_ago"].div(
        df["canteen_week_ago"].max()
    )
    df["canteen_day_ago"] = df["canteen_day_ago"].div(
        df["canteen_day_ago"].max()
    )
    return df.fillna(0)


def preprocess_for_ml(df):
    from_intervall_to_col(df, "Monday", "weekday")
    from_intervall_to_col(df, "Tuesday", "weekday")
    from_intervall_to_col(df, "Wednesday", "weekday")
    from_intervall_to_col(df, "Thursday", "weekday")
    from_intervall_to_col(df, "Friday", "weekday")
    from_intervall_to_col(df, "Saturday", "weekday")
    from_intervall_to_col(df, "Sunday", "weekday")

    df = df.drop(["weekday"], axis=1)
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
    dataframe = dataframe.drop(
        ["max_temp", "min_temp", "precipitation"], axis=1
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
