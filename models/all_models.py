import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_absolute_error

# All the created prediction models
from models.prophet.prophet_model import (
    prophet_predict_canteen_values,
    prophet,
    prophet_create_and_save_model,
)
from models.simple_time_series.simple_time_series import (
    sts_predict_canteen_values,
    simple_time_series,
    create_simple_time_series_model,
)
from models.feed_forward.feed_forward import (
    predict_canteen_values,
    feed_forward_create_model,
)
from models.linear_regression.linear_regression import (
    linear,
    linear_create_model,
)
from models.catboost_model.catboost_model import (
    catboost_predict_values,
    catboost_create_model,
)
from models.lstm.lstm import (
    predict_future_with_trained_model_file,
    lstm_create_model,
)
from helpers.helpers import (
    plot_history_df,
    plot_history,
    load_model_sav,
    plot_prediction,
    plot_history_and_prediction_df,
    plot_history_and_prediction_ml,
)

from preprocessing_df.preprocessing import save_dataframes_next_days
from preprocessing_df.canteen_tail.canteen_tail import (
    get_correlation_historic_canteen_data,
    get_correlation_historic_canteen_no_weekend,
)
from preprocessing.weather.categorize_weather import (
    get_correlation_weather_canteen,
)
from analysis.parking_and_canteen import get_correlation_parking_canteen
from constants import ROOT_DIR, DAYS_TO_TEST
import warnings

warnings.filterwarnings("ignore")


def load_datafiles():
    dt_df = pd.read_csv("{}/data/decision_tree_df.csv".format(ROOT_DIR))
    ml_df = pd.read_csv("{}/data/ml_df.csv".format(ROOT_DIR))
    dt_df.index = pd.to_datetime(dt_df.pop("date"))
    ml_df.index = pd.to_datetime(ml_df.pop("date"))
    return dt_df, ml_df


def load_next_days():
    dt_next, ml_next = save_dataframes_next_days()
    dt_next = dt_next.drop(["Canteen"], axis=1)
    ml_next = ml_next.drop(["Canteen"], axis=1)
    return dt_next, ml_next


def get_correlation():
    return get_correlation_parking_canteen()


def display_canteen_data():
    df = pd.read_csv("{}/data/decision_tree_df.csv".format(ROOT_DIR))
    df.index = pd.to_datetime(df.pop("date"))
    df = df.filter(["Canteen"])

    plt.figure(figsize=(14, 7))
    plt.plot(df)
    plt.title("Number of people at Telenor Oct 2016 - Feb 2019")
    plt.xlabel("Time")
    plt.ylabel("Number of people")


def plot_linear(x, y, x_test, y_pred):
    plt.figure(figsize=(14, 7))
    plt.scatter(x, y, color="black", s=1)
    plt.plot(x_test, y_pred, color="red", linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Number of people")
    plt.legend(["Linear regression trend", "Real values"])


def print_mae(ml_df, filename):
    temp_df_models = ml_df.copy()
    temp_df_models.drop(temp_df_models.tail(8).index, inplace=True)

    model = load_model_sav(filename)
    mae = mean_absolute_error(
        np.asarray(model["Canteen"]), np.asarray(model["prediction"])
    )

    print("Testing set Mean Abs Error: {:5.0f} canteen visitors".format(mae))


def create_dataframe_for_comparison(full_df, split_period):
    """
    Using the last number of rows from the df as a df for comparison.
    @input: split_period = number of days to use from the end of the full dataframe

    """
    df = full_df.iloc[-split_period:]

    # Removes the Canteen data from the df and storing it to another data frame
    real_canteen_series = df.pop("Canteen")
    real_canteen = pd.DataFrame(
        real_canteen_series.values,
        index=real_canteen_series.index,
        columns=["Canteen"],
    )

    return real_canteen, df


def create_predictions(
    dt_df,
    ml_df,
    dt_df_test,
    ml_df_test,
    future=True,
    real_canteen=pd.DataFrame,
):
    """
    Create predictions using all models (except linear regression) and merging the results into one dataframe
    :param dt_df: decision tree dataframe
    :param ml_df: machine learning dataframe
    :param dt_df_test: test dataframe for decision tree
    :param ml_df_test: test dataframe for machine learning
    :param future: Optional Boolean. True if we want to predict the future (default), False if not
    :param real_canteen: Optional dataframe. Contains real canteen values if not predicting the future
    :return merged: Dataframe containing all predictions and real canteen values if provided
    """
    # Using the prediction models
    sts = sts_predict_canteen_values(dt_df, dt_df_test, future)
    prophet = prophet_predict_canteen_values(dt_df, dt_df_test, future)
    feed_forward = predict_canteen_values(ml_df, ml_df_test)
    catboost = catboost_predict_values(dt_df, dt_df_test)
    lstm = predict_future_with_trained_model_file(ml_df, ml_df_test)

    # Merging and renaming all the prediction results
    merged = prophet.copy().rename(columns={"predicted_value": "Prophet"})
    merged = pd.merge(merged, feed_forward, left_index=True, right_index=True)
    merged = merged.rename(columns={"predicted_value": "Feed Forward"})
    merged = pd.merge(merged, catboost, left_index=True, right_index=True)
    merged = merged.rename(columns={"predicted_value": "Catboost"})
    merged["LSTM"] = lstm
    merged["STS"] = sts
    if not real_canteen.empty:
        merged = pd.merge(
            merged, real_canteen, left_index=True, right_index=True
        )
        merged = merged.rename(columns={"Canteen": "Real values"})

    return merged


def plot_all_test_predictions(merged):
    """
    Plotting the dataframe containing all prediction results
    :param merged: merged dataframe
    :return: None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(merged)

    plt.xlabel("Time")
    plt.ylabel("Number of people")
    plt.legend(
        [
            "Prophet",
            "Feed Forward",
            "Catboost",
            "LSTM",
            "Simple Time Series",
            "Real canteen values",
        ],
        loc="best",
    )


def create_and_save_models():
    dt_df, ml_df = load_datafiles()
    dt_df, ml_df = dt_df.copy(), ml_df.copy()
    dt_df.drop(dt_df.tail(DAYS_TO_TEST).index, inplace=True)
    ml_df.drop(dt_df.tail(DAYS_TO_TEST).index, inplace=True)

    catboost_create_model(dt_df)
    feed_forward_create_model(ml_df)

    linear_create_model(
        pd.read_csv("{}/data/dataset.csv".format(ROOT_DIR), index_col="date")
    )
    lstm_create_model(ml_df)
    prophet_create_and_save_model(dt_df)
    create_simple_time_series_model(dt_df)
