import requests
import pandas as pd
import xml.etree.ElementTree as et
from xmljson import BadgerFish
import json


def get_weather_forecast():
    data = get_forecast_from_api()
    df = create_dataframe(data)
    return merge_rows_by_date(df)


def get_forecast_from_api():

    # Define endpoint and parameters
    # (lat and lon are for the weather station at Bygdøy used for weather forecast at Fornebu)
    endpoint = "https://api.met.no/weatherapi/locationforecast/1.9/"
    parameters = {"lat": 59.90, "lon": 10.69, "msl": 15}

    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters)

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
    # Creates DataFrame from api data dict. Collecting relevant information only, between 06:00 and 18:00.

    df_cols = [
        "referenceTime",
        "precipitation",
        "max_temperature",
        "min_temperature",
    ]
    df = pd.DataFrame(columns=df_cols)

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
    # Merging the rows for the same date and summing precipitation, and finding max and min temperature by aggregation

    d = {
        "precipitation": "sum",
        "max_temperature": "max",
        "min_temperature": "min",
    }

    return df.groupby(df.index.date).aggregate(d)