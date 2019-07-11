import pandas as pd
import matplotlib.pyplot as plt

# All the created prediction models
from models.prophet.prophet_model import prophet_predict_canteen_values
from models.simple_time_series.simple_time_series import (
    sts_predict_canteen_values,
)
from models.feed_forward.feed_forward import predict_canteen_values
from models.catboost_model.catboost_model import catboost_predict_values
from models.lstm.lstm import predict_future_with_trained_model_file

# from fbprophet.plot import plot_cross_validation_metric


def load_datafiles():
    dt_df = pd.read_csv("../data/decision_tree_df.csv")
    ml_df = pd.read_csv("../data/ml_df.csv")

    return dt_df, ml_df


def plot_linear(x, y, x_test, y_pred):
    plt.figure(figsize=(14, 7))
    plt.scatter(x, y, color="black", s=1)
    plt.plot(x_test, y_pred, color="red", linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Number of people")
    plt.legend(["Real values", "Linear regression trend"])


def create_dataframe_for_comparison(full_df, split_period):
    """
    Using the last number of rows from the df as a df for comparison.
    @input: split_period = number of days to use from the end of the full dataframe

    """
    df = full_df.iloc[-split_period:]
    df.index = pd.to_datetime(df.pop("date"))

    # Removes the Canteen data from the df and storing it to another data frame
    real_canteen_series = df.pop("Canteen")
    real_canteen = pd.DataFrame(
        real_canteen_series.values,
        index=real_canteen_series.index,
        columns=["Canteen"],
    )

    return real_canteen, df


def create_predictions(dt_df, dt_df_test, ml_df_test):
    sts = sts_predict_canteen_values(dt_df, dt_df_test)
    prophet = prophet_predict_canteen_values(dt_df, dt_df_test)
    feed_forward = predict_canteen_values(ml_df_test)
    catboost = catboost_predict_values(
        dt_df_test, "../data/decision_tree_df.csv"
    )
    history, lstm = predict_future_with_trained_model_file(ml_df_test)

    merged = prophet.copy().rename(columns={"predicted_value": "Prophet"})
    merged = pd.merge(merged, feed_forward, left_index=True, right_index=True)
    merged = merged.rename(columns={"predicted_value": "Feed Forward"})
    merged = pd.merge(merged, catboost, left_index=True, right_index=True)
    merged = merged.rename(columns={"predicted_value": "Catboost"})
    merged["LSTM"] = lstm
    merged["STS"] = sts
    return merged


def plot_all_test_predictions(real_canteen, merged):
    plt.figure(figsize=(16, 8))
    plt.plot(real_canteen)
    plt.plot(merged)

    plt.xlabel("Time")
    plt.ylabel("Number of people")
    plt.legend(
        [
            "Real canteen values",
            "Prophet",
            "Feed Forward",
            "Catboost",
            "LSTM",
            "Simple Time Series Model",
        ],
        loc="best",
    )