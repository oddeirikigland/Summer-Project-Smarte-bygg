import pandas as pd

from preprocessing.canteen_tail.canteen_tail import add_canteen_history
from preprocessing.decision_tree.decision_tree_preprocessing import (
    get_dataset_with_weekday,
)
from preprocessing.start_of_year.start_of_year import add_diff_from_start_year
from preprocessing.weather.categorize_weather import categorize_temperature


def from_intervall_to_col(df, new_col, old_col):
    df[new_col] = df.apply(
        lambda row: 1.0 if row[old_col] == new_col else 0.0, axis=1
    )
    return df


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
    df = preprocess_data(df)

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


def save_dataframes(dataframe):
    df = dataframe.copy()
    decision_tree_df = preprocess_data(df)
    ml_df = preprocess_for_ml(df)
    decision_tree_df.to_csv(
        index=True, path_or_buf="../data/decision_tree_df.csv"
    )
    ml_df.to_csv(index=True, path_or_buf="../data/ml_df.csv")


def main():
    dataframe = pd.read_csv("../data/dataset.csv", index_col="date")
    save_dataframes(dataframe)


if __name__ == "__main__":
    main()
