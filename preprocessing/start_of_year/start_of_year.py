import pandas as pd
from constants import ROOT_DIR


def is_leap_year(year):
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False


def normalize_diff(row):
    denominator = 366 if is_leap_year(row.name.year) else 365
    return row.name.timetuple().tm_yday / denominator


def add_diff_from_start_year(dataframe):
    df = dataframe.copy()
    df.index = pd.to_datetime(df.index)
    df["dist_start_year"] = df.apply(lambda row: normalize_diff(row), axis=1)
    return df


def main():
    df = pd.read_csv("{}/data/dataset.csv".format(ROOT_DIR), index_col="date")
    res = add_diff_from_start_year(df)
    print(res.head())


if __name__ == "__main__":
    main()
