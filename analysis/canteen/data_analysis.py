import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../helpers")
from helpers import all_paths_in_dir


def clean_kdr_data():
    kdr_files = all_paths_in_dir("../../data/kdr_transactions", ".xlsx")

    all_frames = []
    for kdr in kdr_files:
        kdr = pd.read_excel(kdr)
        if len(kdr.columns) > 4:
            kdr = kdr.iloc[:, 0:4]
        kdr.columns = ["Date", "Place", "Menu", "Price"]
        kdr["Date"] = kdr["Date"].astype(str)
        kdr = kdr.iloc[1:]
        kdr = kdr.replace(to_replace=r"(.\d{3})$", value="", regex=True)
        kdr = kdr.replace(to_replace=r"(Selv.*)$", value="", regex=True)
        kdr = kdr.replace(to_replace=r"^(.*FBU)", value="", regex=True)
        kdr = kdr.replace(
            to_replace=r"( \d{2}:\d{2}:\d{2})$", value="", regex=True
        )
        pattern = r"Fresh4You|Soup&Sandwich|eattheStreet"
        filtered_pd = kdr["Place"].str.contains(pattern)
        kdr = kdr[filtered_pd]
        all_frames.append(kdr)
    final_frame = pd.concat(all_frames)
    return final_frame


def amount_location(final_frame):
    amount_per_date = final_frame.groupby(["Date", "Place"])["Place"].count()
    amount_per_date = amount_per_date.to_frame()
    amount_per_date.columns = ["Freq"]
    return amount_per_date


def amount_total(final_frame):
    amount_work = final_frame.groupby("Date")["Place"].count()
    amount_work = amount_work.to_frame()
    amount_work.columns = ["Freq"]

    # Dropping outlier data
    amount_work = amount_work.drop(["2019-01-04"])
    amount_work = amount_work.drop(["2019-01-07"])

    return amount_work


def main():
    final = clean_kdr_data()
    amount_location(final)
    amount_total(final)


if __name__ == "__main__":
    main()
