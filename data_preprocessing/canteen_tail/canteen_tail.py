import pandas as pd
import seaborn as sns
from datetime import timedelta
from constants import ROOT_DIR


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


def get_canteen_values():
    df = pd.read_csv("{}/data/dataset.csv".format(ROOT_DIR))
    df = df.copy()
    df.index = pd.to_datetime(df.pop("date"))
    return df[["Canteen"]]


def get_correlation_historic_canteen_data():
    df = add_canteen_history(get_canteen_values())
    df = df[(df.canteen_week_ago != -1.0) & (df.canteen_day_ago != -1.0)]
    print(
        "Correlation between people eating in canteen with previous day: {0:.2f}".format(
            df["Canteen"].corr(df["canteen_day_ago"])
        )
    )
    print(
        "Correlation between people eating in canteen with one week ago: {0:.2f}".format(
            df["Canteen"].corr(df["canteen_week_ago"])
        )
    )
    df = df.copy().rename(
        columns={
            "Canteen": "Canteen visitors",
            "canteen_day_ago": "Canteen visitors yesterday",
            "canteen_week_ago": "Canteen visitors week ago",
        }
    )
    sns.pairplot(
        df[
            [
                "Canteen visitors",
                "Canteen visitors yesterday",
                "Canteen visitors week ago",
            ]
        ],
        diag_kind="kde",
    )


def get_correlation_historic_canteen_no_weekend():
    dt_df = pd.read_csv("{}/data/decision_tree_df.csv".format(ROOT_DIR))
    dt_df = dt_df.copy()
    dt_df.index = pd.to_datetime(dt_df.pop("date"))
    dt_df = dt_df[
        dt_df.weekday.isin(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        )
    ]
    dt_df = dt_df[["weekday"]]

    df = get_canteen_values()
    canteen_values_no_weekend = pd.merge(
        dt_df, df, left_index=True, right_index=True, how="inner"
    )
    df = add_canteen_history(canteen_values_no_weekend)
    df = df[(df.canteen_week_ago != -1.0) & (df.canteen_day_ago != -1.0)]
    print(
        "Correlation between people eating in canteen with previous day without weekends: {0:.2f}".format(
            df["Canteen"].corr(df["canteen_day_ago"])
        )
    )
    df = df.copy().rename(
        columns={
            "Canteen": "Canteen visitors",
            "canteen_day_ago": "Canteen visitors yesterday",
        }
    )
    sns.pairplot(
        df[["Canteen visitors", "Canteen visitors yesterday"]], diag_kind="kde"
    )


def main():
    get_correlation_historic_canteen_data()
    get_correlation_historic_canteen_no_weekend()


if __name__ == "__main__":
    main()
