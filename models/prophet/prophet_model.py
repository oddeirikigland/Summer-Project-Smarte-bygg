import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics

pd.options.mode.chained_assignment = None  # default='warn'


def preprocess_dataframe(df):
    # Prophet needs data on a specific format
    df.reset_index(inplace=True)
    return df.rename(columns={"date": "ds", "Canteen": "y"})


def prophet_prediction(train, test):
    model = create_model(train)

    # Create dataframe for prediction
    future = model.make_future_dataframe(periods=test.shape[0])
    future["floor"] = 0
    future["cap"] = 2100

    # Predict the future
    forecast = model.predict(future)
    return forecast, model


def create_model(train):
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

    return m


def evaluate_model(model):
    df_cv = cross_validation(
        model, initial="750 days", period="92 days", horizon="8 days"
    )
    df_p = performance_metrics(df_cv)
    return df_cv, df_p


def plot_forecast_and_components(model, forecast):
    model.plot(forecast)
    model.plot_components(forecast)


def get_readable_forecast_info(forecast):
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
