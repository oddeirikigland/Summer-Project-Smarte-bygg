import pandas as pd
import requests
from datetime import date, timedelta
import json
import numpy as np


def get_holiday_for_year(year):
    """
    Get all norwegian holidays for a given year
    :param year: int, year for holidays
    :return: dataframe with holiday information
    """
    response = requests.get("https://webapi.no/api/v1/holidays/" + str(year))

    # Check if the request worked, print out any errors
    if response.status_code == 200:
        res = response.json()
        data = res["data"]

        df = pd.read_json(json.dumps(data))
        df = df.set_index("date")
        return df

    else:
        print("Error! Returned status code %s" % response.status_code)


def calendar_dates_for_year(for_year):
    """
    Gets all dates and weekends for a year to use in dataframe as index.
    :param for_year: int, year you want dates from
    :return: dataframe with year as index and weekends.
    """
    # Source: https://stackoverflow.com/questions/7274267/print-all-day-dates-between-two-dates
    d1 = date(int(for_year), 1, 1)  # start date
    d2 = date(int(for_year), 12, 31)  # end date
    delta = d2 - d1  # timedelta
    dates = {}
    for i in range(delta.days + 1):
        dt = d1 + timedelta(days=i)
        weekend = False
        if dt.weekday() >= 5:
            weekend = True
        dates[dt] = dt.isocalendar()[1], weekend
    return pd.DataFrame.from_dict(dates, orient="index")


def christmas_period(for_year):
    """
    Used for christmas period as we want to set this time as vacation. This is because
        some dates are neither holiday nor 'inneklemt', so it is natural to set these days
        as vacation
    :param for_year: int, year you want dates from
    :return: returns all christmas dates
    """
    # Source: https://stackoverflow.com/questions/7274267/print-all-day-dates-between-two-dates
    d1 = date(int(for_year), 12, 24)  # start date
    d2 = date(int(for_year), 12, 31)  # end date
    delta = d2 - d1  # timedelta
    dates = {}
    for i in range(delta.days + 1):
        dt = d1 + timedelta(days=i)
        dates[dt] = dt.isocalendar()[1]
    return dates


def create_dataframe(year):
    """
    Should create a dataframe with all information (vacation, holidays and 'inneklemt')
    :param year: int, year for which the dataframe should be created
    :return:
    """
    holidays = get_holiday_for_year(year)
    holidays["description"] = True

    holidays.columns = ["holiday"]

    calendar = calendar_dates_for_year(year)
    calendar.columns = ["week", "weekend"]
    calendar.index.name = "date"

    df = pd.merge(
        calendar, holidays, how="outer", left_index=True, right_index=True
    )
    df = df.fillna(False)

    df["vacation"] = False
    # Vinterferie Oslo/Akershus
    df.loc[df.week == 8, "vacation"] = True

    # Høstferie Oslo/Akershus
    df.loc[df.week == 40, "vacation"] = True

    # Fellesferie 3 siste ukene i Juli
    # Assumption
    df.loc[df.week == 28, "vacation"] = True
    df.loc[df.week == 29, "vacation"] = True
    df.loc[df.week == 30, "vacation"] = True
    df.loc[df.weekend, "holiday"] = True if True else False
    for christmas_date in christmas_period(year):
        if not df.loc[christmas_date]["holiday"]:
            df.loc[christmas_date, "vacation"] = True
    df = df.drop(["weekend"], axis=1)
    vacations = find_other_vacation_days(df)
    df["inneklemt"] = vacations
    df = df.drop(["week"], axis=1)
    return df


def find_other_vacation_days(df):
    """
    Algorithm for finding an 'inneklemt dag'. Which is defined as a working day between two free days
        (holiday or weekend).
    :param df: a dataframe with vacations, holidays and weekends as columns
    :return: array of True/False that matches the length of the dataframe given as input
    """
    holidays = df["holiday"].to_numpy()
    holiday = -10
    i = 0
    vacation_days = []
    while i < len(holidays) - 1:
        bool_temp_value = False
        if holidays[i]:
            holiday = i
        else:
            if holiday >= i - 1:
                if holidays[i + 1]:
                    bool_temp_value = True
        vacation_days.append(bool_temp_value)
        i += 1
    vacation_days.append(True)

    return np.array(vacation_days)


def main():
    df = create_dataframe(2017)
    print(df.tail())


if __name__ == "__main__":
    main()
