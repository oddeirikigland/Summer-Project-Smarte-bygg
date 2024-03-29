"""

This script utilize the correlation between the number of people who use the
parking lot and the number of people who eat in the cafeteria. Dates without
canteen data will be updated with data from the parking lot.
"""


import pandas as pd
import numpy as np
import math
from scipy import stats
from analysis.parking.parking import get_cars_parked
from analysis.canteen.data_analysis import amount_total
import seaborn as sns
from constants import ROOT_DIR


def load_and_filter():
    cars_parked = get_cars_parked(
        "{}/data/parking_data".format(ROOT_DIR), ".xlsx"
    )
    canteen = amount_total(
        "{}/data/kdr_transactions".format(ROOT_DIR), ".xlsx"
    )
    canteen.index = pd.to_datetime(canteen.index)
    canteen = canteen.rename(columns={"Freq": "Canteen"})
    return cars_parked, canteen


def remove_outlier(cars_parked, canteen):
    combined = pd.merge(
        cars_parked, canteen, left_index=True, right_index=True, how="inner"
    )
    combined["Proportion"] = combined.apply(
        lambda row: row["Canteen"] / row["Number of cars"], axis=1
    )
    return combined[(np.abs(stats.zscore(combined)) < 3).all(axis=1)]


def update_canteen_column(without_outlier, cars_parked, canteen):
    mean_without_outlier = without_outlier["Proportion"].mean()
    combined_all = pd.merge(
        cars_parked, canteen, left_index=True, right_index=True, how="outer"
    )
    combined_all["Canteen"] = combined_all.apply(
        lambda row: row["Canteen"]
        if not math.isnan(row["Canteen"])
        else row["Number of cars"] * mean_without_outlier,
        axis=1,
    )
    return combined_all


def get_extended_canteen_data():
    cars_parked, canteen = load_and_filter()
    without_outlier = remove_outlier(cars_parked, canteen)
    without_outlier.to_csv(
        index=True,
        path_or_buf="{}/data/raw_canteen_parking.csv".format(ROOT_DIR),
    )
    combined_all = update_canteen_column(without_outlier, cars_parked, canteen)
    combined_all = combined_all.drop(["Number of cars"], axis=1)
    return combined_all


def get_correlation_parking_canteen():
    df = pd.read_csv("{}/data/raw_canteen_parking.csv".format(ROOT_DIR))
    print(
        "Correlation between parked cars and people eating in canteen: {0:.2f}".format(
            df["Canteen"].corr(df["Number of cars"])
        )
    )
    df = df.copy().rename(
        columns={
            "Number of cars": "Cars parked",
            "Canteen": "Canteen visitors",
        }
    )
    sns.pairplot(df[["Cars parked", "Canteen visitors"]], diag_kind="kde")


def main():
    result = get_extended_canteen_data()
    print(result.head())


if __name__ == "__main__":
    main()
