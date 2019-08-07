import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from helpers.helpers import save_model, is_model_saved, load_model_sav
from datetime import datetime
import warnings
import logging

pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore")
logging.getLogger("fbprophet").setLevel(logging.WARNING)


def prophet(df):
    """
    Method to be used in all_models.ipynb for creating prediction and printing graph and MAE
    :param df: input dataset
    :return: None
    """
    df = preprocess_dataframe(df)

    test_period = 8
    train = df.iloc[:-test_period]

    if is_model_saved("prophet.sav"):
        model = load_model_sav("prophet")
    else:
        model = create_prophet_model(train)
    forecast = prediction(model, test_period)
    model.plot(forecast, xlabel="Date", ylabel="Number of people")
    df_cv, df_p = evaluate_model(model)

    print(
        "The mean absolute error (MAE) for the Prophet model is {0:.0f} people".format(
            df_p["mae"].mean()
        )
    )


def prophet_predict_canteen_values(df, prediction_df, future=True):
    """
    Returns the predicted values for the prediction_df
    :param df: full dataframe
    :param prediction_df: the prediction dataframe (where to predict number of people)
    :param future: Optional Boolean. True if we want to predict the future (default), False if not
    :return: The predicted values for the dates in prediction_df
    """
    df = preprocess_dataframe(df)
    test_period = prediction_df.shape[
        0
    ]  # number of days we are predicting (possibly in the future)

    # The days between last row in dataset and today needs to be calculated
    if future is True:
        date_today = datetime.now()
        end_date = df["ds"].iloc[-1]
        test_period = test_period + (date_today - end_date).days - 1
        # The whole dataframe is now the train set
        train = df
    else:
        train = df.iloc[:-test_period]

    model = create_prophet_model(train)
    forecast = prediction(model, test_period)
    # Prophet returns alot as a forecast, but we are only interested in date and predicted number of people
    filtered_forecast = forecast.filter(["ds", "yhat"])
    renamed = filtered_forecast.rename(
        columns={"ds": "date", "yhat": "predicted_value"}
    )
    renamed.index = pd.to_datetime(renamed.pop("date"))

    return renamed.iloc[-prediction_df.shape[0] :]


def preprocess_dataframe(in_df):
    # Creating a copy, setting index as datetime and only keeping date and Canteen
    df = in_df.copy()
    df = df.asfreq("D")
    df = df.filter(["date", "Canteen"])
    df.fillna(method="ffill", inplace=True)

    # Prophet needs data on a specific format: ds and y
    df.reset_index(inplace=True)
    return df.rename(columns={"date": "ds", "Canteen": "y"})


def prediction(model, test_period):
    """
    Creating prediction.
    :param model: trained model
    :param test_period: number of days to predict
    :return: prediction/forecast
    """
    # Create dataframe for prediction and setting max and min values for the data
    future = model.make_future_dataframe(periods=test_period)
    future["floor"] = 0
    future["cap"] = 2100

    # Predict the future
    forecast = model.predict(future)
    return forecast


def create_prophet_model(train):
    # Create model based on training dataframe
    train["floor"] = 0
    train["cap"] = 2100

    m = Prophet(
        growth="logistic",
        interval_width=0.95,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=20,
        seasonality_prior_scale=20,
        yearly_seasonality=12,
        daily_seasonality=False,
    )
    m.add_seasonality(name="monthly", period=30.5, fourier_order=4)
    m.add_country_holidays(country_name="Norway")
    m.fit(train)
    # m.train_holiday_names
    save_model(m, "prophet")
    return m


def evaluate_model(model):
    df_cv = cross_validation(
        model, initial="700 days", period="92 days", horizon="8 days"
    )
    df_p = performance_metrics(df_cv)
    return df_cv, df_p


def plot_forecast_and_components(model, forecast):
    model.plot(forecast)
    model.plot_components(forecast)


def get_readable_forecast_info(forecast):
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def prophet_create_and_save_model(dt_df):
    dt_df = preprocess_dataframe(dt_df)
    create_prophet_model(dt_df)


def main():
    dt_df = pd.read_csv("../../data/decision_tree_df.csv")
    dt_df_test = dt_df.iloc[-8:]
    dt_df_test.index = pd.to_datetime(dt_df_test.pop("date"))

    canteen_prediction = prophet_predict_canteen_values(dt_df_test)
    print(canteen_prediction)


if __name__ == "__main__":
    main()
