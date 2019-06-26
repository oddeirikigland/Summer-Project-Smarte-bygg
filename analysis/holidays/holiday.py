import pandas as pd
import requests
from datetime import date, timedelta
import json
import numpy as np


def get_holiday_for_year(year):
    response = requests.get("https://webapi.no/api/v1/holidays/" + str(year))

    res = response.json()

    # Check if the request worked, print out any errors
    if response.status_code == 200:
        data = res["data"]
        print("Data retrieved from web api!")

        df = pd.read_json(json.dumps(data))
        df = df.set_index("date")
        return df

    else:
        print("Error! Returned status code %s" % response.status_code)
        print("Message: %s" % res["error"]["message"])
        print("Reason: %s" % res["error"]["reason"])


def calendar_dates_for_year(for_year):
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


def create_dataframe(year):
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
    # Might need to change this assumption?
    df.loc[df.week == 28, "vacation"] = True
    df.loc[df.week == 29, "vacation"] = True
    df.loc[df.week == 30, "vacation"] = True

    df.loc[df.weekend, "holiday"] = True if True else False
    df = df.drop(["weekend"], axis=1)
    vacations = find_other_vacation_days(df)
    df["vacation2"] = vacations

    df.loc[df.vacation2, "vacation"] = True if True else False
    df = df.drop(["vacation2"], axis=1)
    return df


def find_other_vacation_days(df):
    holidays = df["holiday"].to_numpy()
    holiday = -10
    i = 0
    vacation_days = []
    while i < len(holidays) - 2:
        bool_temp_value = False
        if holidays[i]:
            holiday = i
        else:
            if holiday >= i - 1:
                if holidays[i + 1] or holidays[i + 2]:
                    bool_temp_value = True
            if holiday >= i - 2:
                if holidays[i + 1]:
                    bool_temp_value = True
        vacation_days.append(bool_temp_value)
        i += 1
    vacation_days.append(True)
    vacation_days.append(True)
    return np.array(vacation_days)


def main():
    create_dataframe(2018)


if __name__ == "__main__":
    main()
