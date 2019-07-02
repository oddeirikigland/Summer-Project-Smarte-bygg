import pandas as pd
from datetime import timedelta


def previous_day_to_col(df, row, days):
    x_days_earlier = row.name - timedelta(days=days)
    return df.Canteen.get(x_days_earlier, -1)


def add_canteen_history(dataframe):
    df = dataframe.copy()
    df.index = pd.to_datetime(df.index)
    df["canteen_week_ago"] = df.apply(
        lambda row: previous_day_to_col(df, row, 7), axis=1
    )
    df["canteen_day_ago"] = df.apply(
        lambda row: previous_day_to_col(df, row, 1), axis=1
    )
    return df


def main():
    df = pd.read_csv("../../data/dataset.csv", index_col="date")
    res = add_canteen_history(df)
    print(res.head())


if __name__ == "__main__":
    main()
