import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from statsmodels.tsa.stattools import adfuller
from helpers.helpers import save_model
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


def simple_time_series(full_df, test_period, display_graphs=True):
    """
    Creating prediction and displaying graph and MAE. To be used in all_models.
    :param full_df: full dataframe
    :param test_period: Length/size of the test set
    :param display_graphs: Display graph if only if True
    :return: None
    """
    df = full_df.copy()
    df.index = pd.to_datetime(df.pop("date"))
    df = df.filter(["Canteen"])

    train = df.iloc[:-test_period]
    test = df.iloc[-test_period:]

    resulting_prediction, predictions = prediction(train, test)

    if display_graphs is True:
        plt.figure(figsize=(14, 7))
        plt.plot(train)
        plt.plot(resulting_prediction)
        plt.legend(["Real values", "Prediction"], loc="best")
        plt.xlabel("Time")
        plt.ylabel("Number of people")

        print(
            "The mean absolute error (MAE) for the Simple Time Series model is {0:.0f} people".format(
                find_MAE(test, predictions)
            )
        )


def sts_predict_canteen_values(full_df, prediction_df, future=True):
    """
    Returns the predicted values for the prediction_df
    :param full_df: full dataframe
    :param prediction_df: the prediction dataframe
    :param future: Optional Boolean. True if we want to predict the future (default), False if not
    :return: The predicted values for the dates in prediction_df
    """

    df = full_df.copy()
    df.index = pd.to_datetime(df.pop("date"))
    df = df.filter(["Canteen"])
    test_period = prediction_df.shape[0]

    # The days between last row in dataset and today needs to be calculated
    if future is True:
        # Finding days between end date of dataset and today
        date_today = datetime.now()
        end_date = df.index[-1]
        future_test_period = test_period + (date_today - end_date).days - 1

        # The whole dataframe is now the test set
        train = df
        # Creating a new dataframe containing all dates between end date and today + future days from prediction_df
        date_df = pd.DataFrame()
        date_df["date"] = pd.date_range(
            (end_date + timedelta(1)).date(),
            periods=future_test_period,
            freq="D",
        )
        date_df.index = pd.to_datetime(date_df.pop("date"))
        # Setting the new df as test set
        test = date_df
    else:
        # If future = False, the test period is the last part of the full_df, train is everything up until this point
        train = df.iloc[:-test_period]
        test = df.iloc[-test_period:]

    # Make prediction
    _, predictions = prediction(train, test)
    return predictions.iloc[-test_period:]


def prediction(train, test):
    """
    Make prediction. The code is based on the following article:
    https://numbersandcode.com/another-simple-time-series-model-using-naive-bayes-for-forecasting
    :param train: train dataframe
    :param test: test dataframe
    :return: prediction for training dataset, prediction for test dataset
    """
    # Binning the data: To bin the data, intervals are being assigned.
    # Now, each continuous observation xt is replaced by an indicator xt=k, where k is the interval that xt falls in.
    # The number of intervals was chosen arbitrarily.
    bins = [-1, 1, 200, 500, 1000, 1500, 1700, 1800, 1900, 10000]
    binned_series, bin_means = bin_data(train, bins)

    # Getting the lagged lists for the regressor (train_x) and regressand (train_y)
    train_x, train_y = get_lagged_list(binned_series, test.shape[0])

    # Create model
    model = create_sts_model(train_x, train_y)
    # Create prediction for train dataset
    resulting_prediction = find_training_prediction(
        train_x, train_y, model, bin_means
    )
    # Create prediction for test dataset
    predictions, pred_class = find_prediction_forecast(
        test, train_x, train_y, model, bin_means
    )

    return resulting_prediction, predictions


def bin_data(dataset, bins):
    """
    The data are binned and the mean of realizations xt in each interval is saved in a dictionary in order to
    map the interval category back to actual realizations (bin_means).
    :param dataset: the dataset (dataframe) that are to be binned
    :param bins: list of numbers (= intervals between the numbers)
    :return: a series of the binned data and the bin means
    """
    binned = np.digitize(dataset.iloc[:, 0], bins)
    bin_means = {}

    for binn in range(1, len(bins)):
        bin_means[binn] = dataset[binned == binn].mean()

    return pd.Series(binned, index=dataset.index), bin_means


def get_lagged_list(binned_series, lags):
    """
    To forecast future realizations, the classic approach of using lagged realizations of xt will be applied.
    :param binned_series: the data in binned series
    :param lags: number of lags to use. This will typically be the size of the test set
    :return: two lagged lists: regressor (train_x) and regressands (train_y)
    """
    lagged_list = []
    for s in range(lags):
        lagged_list.append(binned_series.shift(s))

    lagged_frame = pd.concat(lagged_list, 1).dropna()

    train_x = lagged_frame.iloc[:, 1:]
    train_y = lagged_frame.iloc[:, 0]
    return train_x, train_y


def create_sts_model(train_x, train_y):
    """
    Create model using Gaussian Naive Bayes and save this
    :param train_x: lagged list, regressor
    :param train_y: lagged_list, regressand
    :return: trained model
    """
    model = GaussianNB()
    model.fit(train_x, train_y)
    save_model(model, "simple_time_series")
    return model


def find_training_prediction(train_x, train_y, model, bin_means):
    """
    Returns the prediction for the training set.
    :param train_x: lagged list, regressor
    :param train_y: lagged_list, regressand
    :param model: trained model
    :param bin_means: the means from the bins
    :return: prediction of train set
    """
    # Predicted bin values
    pred_insample = model.predict(train_x)
    pred_insample = pd.DataFrame(pred_insample, index=train_y.index)

    resulting_prediction = pd.Series(np.nan, index=train_y.index)
    for row in range(len(pred_insample)):
        # The resulting prediction is equal to the means for the predicted bin
        mean_class = get_mean_from_class(pred_insample.values[row], bin_means)
        resulting_prediction.iloc[row] = mean_class[0]

    return resulting_prediction


def find_prediction_forecast(test, train_x, train_y, model, bin_means):
    """
    Returns the prediction for a test set (out of sample forecast), both predicted numbers and classes.
    :param test: test dataframe
    :param train_x: lagged list, regressor
    :param train_y: lagged_list, regressand
    :param model: trained model
    :param bin_means: the means from the bins
    :return: prediction of test dataset
    """
    prediction_frame = pd.DataFrame(
        np.nan, index=test.index, columns=range(train_x.shape[1])
    )
    predictions = pd.Series(index=test.index)
    pred_class = pd.Series(index=test.index)

    prediction_frame.iloc[0, 1:] = train_x.iloc[-1, :-1].values
    prediction_frame.iloc[0, 0] = train_y.iloc[-1]

    # Out-of-sample forecasts need to be calculated iteratively since lagged values are required.
    for i in range(len(test)):
        pred = model.predict(prediction_frame.iloc[i, :].values.reshape(1, -1))
        pred_class.iloc[i] = pred
        predictions.iloc[i] = get_mean_from_class(pred.reshape(-1), bin_means)[
            0
        ]
        try:
            prediction_frame.iloc[i + 1, 1:] = prediction_frame.iloc[
                i, :-1
            ].values
            prediction_frame.iloc[i + 1, 0] = pred[0]
        except IndexError:
            pass

    return predictions, pred_class.astype("int")


def get_mean_from_class(prediction, bin_means):
    return bin_means[prediction[0]]


def test_stationarity(dataframe, time_window):
    """
    Test if the timeseries in dataframe is stationary
    :param dataframe:
    :param time_window:
    :return:
    """
    timeseries = dataframe.iloc[:, 0]
    # Determing rolling statistics
    rolmean = timeseries.rolling(time_window).mean()
    rolstd = timeseries.rolling(time_window).std()

    # Plot rolling statistics:
    plt.figure(figsize=(18, 9))
    plt.plot(timeseries, color="blue", label="Original")
    plt.plot(rolmean, color="red", label="Rolling Mean")
    plt.plot(rolstd, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean & Standard Deviation")
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)


def test_accuracy(pred_class, binned_test_series):
    # Checks if the model predicts the same categories as the actual ones
    testing_size = binned_test_series.shape[0]

    # Counts number of times the prediction is correct
    comparison = binned_test_series == pred_class
    counts = comparison.value_counts()[True]

    accuracy = counts / testing_size * 100
    return accuracy


def find_MAE(dataset, prediction):
    return np.sqrt(np.mean((dataset.iloc[:, 0] - prediction) ** 2))


def main():
    dt_df = pd.read_csv("../../data/decision_tree_df.csv")
    dt_df_test = dt_df.iloc[-8:]
    dt_df_test.index = pd.to_datetime(dt_df_test.pop("date"))

    canteen_prediction = sts_predict_canteen_values(dt_df_test)
    print(canteen_prediction)


if __name__ == "__main__":
    main()
