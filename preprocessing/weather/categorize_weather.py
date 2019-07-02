import pandas as pd


def replace_temps_with_avg(df):
    df["avg_temp"] = df[["max_temp", "min_temp"]].mean(axis=1)
    df.drop(labels=["max_temp", "min_temp"], axis="columns", inplace=True)
    return df


def categorize_temperature(df):
    """
    Using the following categories:
        Avg temperature:  -10 <= x <= -2 || +2 <= x <= 20(preferred_work_temp),
                          x < -10 || -2 < x < +2 || x > 20 (stay_home_temp)
    """

    # Setting labels and bin intervals
    temp_bins = [
        -20,
        -10,
        -2,
        2,
        20,
        40,
    ]  # The minimum avg temp is -12.75 and max is 27.8

    df["avg_temp"] = pd.cut(
        df["avg_temp"], temp_bins, labels=["a", "b", "c", "d", "e"]
    )
    my_dict = dict.fromkeys(["a", "c", "e"], "stay_home_temp")
    my_dict.update(dict.fromkeys(["b", "d"], "preferred_work_temp"))
    df["avg_temp"] = df["avg_temp"].map(my_dict)

    return df


def main():
    df = pd.read_csv("../../data/dataset.csv")
    categorize_temperature(replace_temps_with_avg(df))


if __name__ == "__main__":
    main()
