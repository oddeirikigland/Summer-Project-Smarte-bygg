import requests
import pandas as pd
import xml.etree.ElementTree as et
from xmljson import BadgerFish
import json
import sys


def get_weather_forecast():
    data = get_forecast_from_api()
    df = create_dataframe(data)
    df = merge_rows_by_date(df)
    return df.rename(
        {"max_temperature": "max_temp", "min_temperature": "min_temp"}, axis=1
    )


def get_forecast_from_api():
    """
    Using api.met.no to get the weather forecast
    :return data: data dictionary with relevant data from the api
    """
    # Define endpoint and parameters
    # (lat and lon are for the weather station at Bygdøy used for weather forecast at Fornebu)
    endpoint = "https://api.met.no/weatherapi/locationforecast/1.9/"
    parameters = {"lat": 59.90, "lon": 10.69, "msl": 15}

    try:
        # Issue an HTTP GET request
        r = requests.get(endpoint, parameters)

    except requests.exceptions.RequestException as e:
        print("Not possible to get the weather forecast from api.met.no")
        print(e)
        sys.exit()

    # Handling the XML response
    bf = BadgerFish()
    json_string = json.dumps(
        bf.data(et.fromstring(r.content))
    )  # creating json string
    json_dict = json.loads(json_string)  # creating json dict
    data = json_dict["weatherdata"]["product"][
        "time"
    ]  # collecting the relevant part of the response

    return data


def create_dataframe(data):
    """
    Creates DataFrame from api data dict. Collecting relevant information only, between 06:00 and 18:00.
    :param data: data dictionary from api
    :return df: dataframe containing time, precipitation, max and min temperatures
    """

    df_cols = [
        "referenceTime",
        "precipitation",
        "max_temperature",
        "min_temperature",
    ]
    df = pd.DataFrame(columns=df_cols)

    # Using the spesific format of the XML response stored in the dict. Only using data between 06:00 and 18:00
    for i in range(len(data)):
        if (
            "T06:00:00Z" in data[i]["@from"] and "T12:00:00Z" in data[i]["@to"]
        ) or (
            "T12:00:00Z" in data[i]["@from"] and "T18:00:00Z" in data[i]["@to"]
        ):
            s_time = data[i]["@from"]
            s_preci = data[i]["location"]["precipitation"]["@value"]
            s_max_temp = data[i]["location"]["maxTemperature"]["@value"]
            s_min_temp = data[i]["location"]["minTemperature"]["@value"]

            df = df.append(
                pd.Series(
                    [s_time, s_preci, s_max_temp, s_min_temp], index=df_cols
                ),
                ignore_index=True,
            )

    # Convert the time value to datetime and set as index
    df.index = pd.to_datetime(df.pop("referenceTime"))

    return df


def merge_rows_by_date(df):
    """
    Merging the rows for the same date and summing precipitation, and finding max and min temperature by aggregation
    :param df: dataframe containing datetime, precipitation, max and min temp
    :return: new dataframe with only one row per date. Aggregated numbers.
    """

    d = {
        "precipitation": "sum",
        "max_temperature": "max",
        "min_temperature": "min",
    }

    return df.groupby(df.index.date).aggregate(d)
