import requests
import pandas as pd
import configparser


def get_weather_data(start_time, end_time, config_path):
    # Getting client id and secret from configparser file
    config = configparser.ConfigParser()
    config.read(config_path)

    client_id = config["frost_client"]["client_id"]
    client_secret = config["frost_client"]["client_secret"]

    # Define endpoint and parameters
    endpoint = "https://frost.met.no/observations/v0.jsonld"
    parameters = {
        "sources": "SN18815",
        "elements": "sum(precipitation_amount PT12H), min(air_temperature PT12H), max(air_temperature PT12H)",
        "referencetime": start_time + "/" + end_time,
    }

    # Issue an HTTP GET request and extract JSON data
    r = requests.get(endpoint, parameters, auth=(client_id, client_secret))
    json = r.json()

    # Check if the request worked, print out any errors
    if r.status_code == 200:
        data = json["data"]
        print("Data retrieved from frost.met.no!")
        return clean_data(data)

    else:
        print("Error! Returned status code %s" % r.status_code)
        print("Message: %s" % json["error"]["message"])
        print("Reason: %s" % json["error"]["reason"])


def clean_data(data):
    # Returns a Dataframe with all of the observations in a table format
    df = pd.DataFrame()
    for i in range(len(data)):
        row = pd.DataFrame(data[i]["observations"])
        row["referenceTime"] = data[i]["referenceTime"]
        row["sourceId"] = data[i]["sourceId"]
        df = df.append(row)

    df = df.reset_index()

    # These columns will be kept
    columns = [
        "sourceId",
        "referenceTime",
        "elementId",
        "value",
        "unit",
        "timeOffset",
    ]
    df2 = df[columns].copy()

    # Convert the time value to datetime and set as index
    df2.index = pd.to_datetime(df2.pop("referenceTime"))
    # Only keeping measurements from 06:00
    return df2.at_time("06:00:00+00:00")


def split_dataframe(df):
    precipitation = df.loc[
        df["elementId"] == "sum(precipitation_amount PT12H)"
    ]
    max_temp = df.loc[df["elementId"] == "max(air_temperature PT12H)"]
    min_temp = df.loc[df["elementId"] == "min(air_temperature PT12H)"]

    return precipitation, max_temp, min_temp


def plot_precipitation(precipitation):
    preci_plot = precipitation.plot(
        y="value", title="Precipitation", legend=False, figsize=(10, 8)
    )
    preci_plot.set_xlabel("Time")
    preci_plot.set_ylabel("Precipitation [mm]")


def plot_temperature(max_temp, min_temp):
    ax = max_temp.plot(y="value")
    temp_plot = min_temp.plot(
        ax=ax, y="value", title="Temperatures", figsize=(10, 8)
    )

    temp_plot.set_xlabel("Time")
    temp_plot.set_ylabel("Temperature [*C]")
    temp_plot.legend(["Max temp", "Min temp"])


def get_split_weather_data(start_time, end_time, config_path):
    result = get_weather_data(start_time, end_time, config_path)
    precipitation, max_df, min_df = split_dataframe(result)
    per = precipitation[["value"]]
    per.columns = ["precipitation"]
    merged = pd.merge(
        per, result, left_index=True, right_index=True, how="inner"
    )

    max_temp = max_df[["value"]]
    max_temp.columns = ["max_temp"]
    merged = pd.merge(
        merged, max_temp, left_index=True, right_index=True, how="inner"
    )

    min_temp = min_df[["value"]]
    min_temp.columns = ["min_temp"]
    merged = pd.merge(
        merged, min_temp, left_index=True, right_index=True, how="inner"
    )

    merged = merged.drop(["value", "elementId", "unit"], axis=1)

    merged = merged.drop_duplicates()

    return merged


def main():
    result = get_weather_data("2018-01-01", "2018-02-15", "../../config.ini")
    print(result)


if __name__ == "__main__":
    main()
