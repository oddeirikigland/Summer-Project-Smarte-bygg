import pandas as pd
from helpers.helpers import load_excel_files_into_df
from constants import ROOT_DIR


# Sorts parking data for duplicate function
def sorted_parking_data(dataframe):
    df = dataframe.copy()
    df.index = pd.to_datetime(df.pop("Starttid"))
    df = df.between_time("04:00", "12:00")
    df["Sluttid"] = pd.to_datetime(df["Sluttid"])
    return df.sort_values(by=["Starttid"])


# Finds number of duplicates in parking data for a given day
def parking_duplicates(sorted_parking_data, date):
    count = 0
    sorted_parking_data = sorted_parking_data[date]
    for i in range(sorted_parking_data.shape[0] - 1):
        time_diff_start = (
            sorted_parking_data.index[i + 1] - sorted_parking_data.index[i]
        )
        if time_diff_start.seconds < 10:
            time_diff_end = (
                sorted_parking_data["Sluttid"][i + 1]
                - sorted_parking_data["Sluttid"][i]
            )
            if time_diff_end.seconds < 10:
                count += 1
    return count


def parking_counter(
    dataframe, new_index, column_name_counter, new_column_name
):
    df = dataframe.copy()
    df.index = pd.to_datetime(df.pop(new_index))
    df = df.between_time("04:00", "12:00")
    df = df.rename(columns={column_name_counter: new_column_name})
    return df.groupby(df.index.date).count()


def number_of_parked_cars(dataframe):
    arrival = parking_counter(dataframe, "Starttid", "Sluttid", "Arrival")
    departure = parking_counter(dataframe, "Sluttid", "Starttid", "Departure")
    res = pd.merge(arrival, departure, left_index=True, right_index=True)

    parking_data = sorted_parking_data(dataframe)
    res["Duplicates"] = res.apply(
        lambda row: parking_duplicates(parking_data, str(row.name)), axis=1
    )
    res["Number of cars"] = (
        res["Arrival"] - res["Departure"] - res["Duplicates"]
    )
    res = res.drop(["Arrival", "Departure", "Duplicates"], axis=1)
    return res


def get_cars_parked(path, file_type):
    df = load_excel_files_into_df(path, file_type)
    dataframe = df.copy()
    dataframe = dataframe[dataframe["Avdeling"].str.contains("P1")]
    dataframe = dataframe.drop(["Avdeling", "Payment status"], axis=1)
    return number_of_parked_cars(dataframe)


def main():
    result = get_cars_parked("{}/data/parking_data".format(ROOT_DIR), ".xlsx")
    print(result.head(20))


if __name__ == "__main__":
    main()
