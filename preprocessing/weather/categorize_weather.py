import pandas as pd
from constants import ROOT_DIR
import seaborn as sns
from preprocessing.decision_tree.decision_tree_preprocessing import (
    get_dataset_with_weekday,
)


def replace_temps_with_avg(df):
    df = df.copy()
    df["avg_temp"] = df[["max_temp", "min_temp"]].mean(axis=1)
    df.drop(labels=["max_temp", "min_temp"], axis="columns", inplace=True)
    return df


def categorize_temperature(df):
    """
    Using the following categories:
        Avg temperature:  x ∈ (−10,−2], x ∈ (2,20] (preferred_work_temp),
                          x ∈ (−∞,−10], x ∈ (−2,2], x ∈ (20,∞) (stay_home_temp)
    """

    df = replace_temps_with_avg(df)
    # Setting labels and bin intervals
    temp_bins = [-20, -10, -2, 2, 20, 40]

    # The minimum avg temp is -12.75 and max is 27.8.
    # -20 and 40 are randomly chosen number smaller or bigger than these

    # Creating bins from the avg_temp and adding random labels to be able to separate into stay_home and preferred_work
    df["avg_temp"] = pd.cut(
        df["avg_temp"], temp_bins, labels=["a", "b", "c", "d", "e"]
    )

    # Creating dictionary and adding the relevant intervals to the two categories
    my_dict = dict.fromkeys(["a", "c", "e"], "stay_home_temp")
    my_dict.update(dict.fromkeys(["b", "d"], "preferred_work_temp"))

    # Mapping the avg_temp values to the dict values (categories)
    df["avg_temp"] = df["avg_temp"].map(my_dict)

    return df


def get_correlation_weather_canteen(df):
    # Creating the correct data format, with categorized temperatures and including week days
    df = categorize_temperature(df)
    # Mapping the categories to numbers
    codes = {"preferred_work_temp": 0, "stay_home_temp": 1}
    df["avg_temp"] = df["avg_temp"].map(codes)

    # Removing holidays
    df = df.loc[df["holiday"] == 0.0]

    # Renaming columns for aesthetic reasons
    df = df.copy().rename(
        columns={
            "Canteen": "Canteen visitors",
            "precipitation": "Precipitation",
            "avg_temp": "Temperature",
        }
    )

    # Temperature
    print(
        "Correlation between people eating in canteen and temperature: {0:.2f}".format(
            abs(df["Canteen visitors"].corr(df["Temperature"]))
        )
    )

    # Precipitation
    print(
        "Correlation between people eating in canteen and precipitation: {0:.2f}".format(
            abs(df["Canteen visitors"].corr(df["Precipitation"]))
        )
    )

    sns.pairplot(
        df[["Canteen visitors", "Temperature", "Precipitation"]],
        diag_kind="kde",
    )


def main():
    df = pd.read_csv("{}/data/dataset.csv".format(ROOT_DIR))
    get_correlation_weather_canteen(df)


if __name__ == "__main__":
    main()
