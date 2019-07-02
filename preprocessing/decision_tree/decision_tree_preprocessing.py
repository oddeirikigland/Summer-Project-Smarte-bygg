import sys
import os
from datetime import datetime, date
import calendar
import pandas as pd


def get_dataset_with_weekday(dataset):
    # Ask Odd Eirik for this solution

    dataset["index1"] = dataset.index
    dataset["index1"] = pd.to_datetime(dataset["index1"])
    dataset["index1"] = dataset.apply(
        lambda row: calendar.day_name[row["index1"].weekday()], axis=1
    )

    dataset.rename(columns={"index1": "weekday"}, inplace=True)

    return dataset


def main():
    sys.path.append(
        os.path.dirname(os.path.realpath(__file__)) + "/../../analysis"
    )
    from combined_dataset import open_csv

    raw_dataset = open_csv("../../data/dataset.csv")
    dataset = raw_dataset.copy()

    df = get_dataset_with_weekday(dataset)
    print(df)


if __name__ == "__main__":
    main()
