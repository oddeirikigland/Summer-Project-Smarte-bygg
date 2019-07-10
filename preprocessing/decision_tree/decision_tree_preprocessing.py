import calendar
import pandas as pd
from constants import ROOT_DIR


def get_dataset_with_weekday(dataset):
    dataset["index1"] = dataset.index
    dataset["index1"] = pd.to_datetime(dataset["index1"])
    dataset["index1"] = dataset.apply(
        lambda row: calendar.day_name[row["index1"].weekday()], axis=1
    )

    dataset.rename(columns={"index1": "weekday"}, inplace=True)

    return dataset


def main():
    dataset = pd.read_csv(
        "{}/data/dataset.csv".format(ROOT_DIR), index_col="date"
    )
    df = get_dataset_with_weekday(dataset)
    print(df)


if __name__ == "__main__":
    main()
