import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from helpers.helpers import save_model

pd.options.mode.chained_assignment = None  # default='warn'


def prophet(df):
    df = preprocess_dataframe(df)

    test_period = 8
    train = df.iloc[:-test_period]

    model = create_prophet_model(train)
    forecast = prediction(model, test_period)
    model.plot(forecast)
    df_cv, df_p = evaluate_model(model)

    print(
        "The mean absolute error (MAE) for the Prophet model is {0:.2f}".format(
            df_p["mae"].mean()
        )
    )


def prophet_predict_canteen_values(df, prediction_df, future=False):
    df = preprocess_dataframe(df)

    # Splitting in test and train datasets
    test_period = prediction_df.shape[
        0
    ]  # days we are predicting in the future
    train = df.iloc[:-test_period]
    # test = df.iloc[-test_period:]

    model = create_prophet_model(train)
    save_model(model, "prophet")
    forecast = prediction(model, test_period)
    filtered_forecast = forecast.filter(["ds", "yhat"])
    renamed = filtered_forecast.rename(
        columns={"ds": "date", "yhat": "predicted_value"}
    )
    renamed.index = pd.to_datetime(renamed.pop("date"))

    return renamed.iloc[-test_period:]


def preprocess_dataframe(in_df):
    df = in_df.copy()
    df.index = pd.to_datetime(df.pop("date"))
    df = df.asfreq("D")
    df = df.filter(["date", "Canteen"])
    df.fillna(method="ffill", inplace=True)

    # Prophet needs data on a specific format
    df.reset_index(inplace=True)
    return df.rename(columns={"date": "ds", "Canteen": "y"})


def prediction(model, test_period):
    # Create dataframe for prediction
    future = model.make_future_dataframe(periods=test_period)
    future["floor"] = 0
    future["cap"] = 2100

    # Predict the future
    forecast = model.predict(future)
    return forecast


def create_prophet_model(train):
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


def main():
    dt_df = pd.read_csv("../../data/decision_tree_df.csv")
    dt_df_test = dt_df.iloc[-8:]
    dt_df_test.index = pd.to_datetime(dt_df_test.pop("date"))

    canteen_prediction = prophet_predict_canteen_values(dt_df_test)
    print(canteen_prediction)


if __name__ == "__main__":
    main()
