import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_preprocessing.decision_tree.decision_tree_preprocessing import (
    get_dataset_with_weekday,
)
from constants import ROOT_DIR


def plot_mean_workers_per_day(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.filter(items=["date", "Canteen"])
    df = get_dataset_with_weekday(df)
    mean_dict = {}
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        day_df = df.copy()
        day_df = day_df[day_df["weekday"] == day]
        mean_dict[day] = day_df["Canteen"].mean()

    y_pos = np.arange(len(mean_dict.keys()))
    plt.figure(figsize=(15, 5))
    plt.bar(y_pos, mean_dict.values(), align="center")
    plt.xticks(y_pos, mean_dict.keys())
    plt.ylabel("People at work")
    plt.show()


def main():
    df = pd.read_csv("{}/data/dataset.csv".format(ROOT_DIR))
    plot_mean_workers_per_day(df)


if __name__ == "__main__":
    main()
