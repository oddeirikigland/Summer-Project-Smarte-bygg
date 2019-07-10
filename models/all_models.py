import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# All the created prediction models
from prophet.prophet_model import *
from simple_time_series.simple_time_series import *
from feed_forward.feed_forward import *
from catboost_model.catboost_model import *
from lstm.lstm import *

# from fbprophet.plot import plot_cross_validation_metric


def load_datafiles():
    dt_df = pd.read_csv("../data/decision_tree_df.csv")
    ml_df = pd.read_csv("../data/ml_df.csv")

    return dt_df, ml_df


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
