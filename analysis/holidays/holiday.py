import pandas as pd
import requests


def get_holiday_for_year(year):
    response = requests.get("https://webapi.no/api/v1/holidays/" + year)

    json = response.json()

    # Check if the request worked, print out any errors
    if response.status_code == 200:
        data = json["data"]
        print("Data retrieved from web api!")
        return data

    else:
        print("Error! Returned status code %s" % response.status_code)
        print("Message: %s" % json["error"]["message"])
        print("Reason: %s" % json["error"]["reason"])


def get_calendar_dates_year(year):
    response = requests.get("https://webapi.no/api/v1/calendar/" + year)

    json = response.json()

    # Check if the request worked, print out any errors
    if response.status_code == 200:
        data = json["data"]
        print("Data retrieved from web api!")
        return data

    else:
        print("Error! Returned status code %s" % response.status_code)
        print("Message: %s" % json["error"]["message"])
        print("Reason: %s" % json["error"]["reason"])


def create_data_frame(calendar):
    for month in calendar["months"]:
        print("hey")


def main():

    print(get_holiday_for_year("2018"))
    calendar_dates = get_calendar_dates_year("2018")
    create_data_frame(calendar_dates)


if __name__ == "__main__":
    main()
