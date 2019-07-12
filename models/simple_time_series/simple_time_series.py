import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from statsmodels.tsa.stattools import adfuller
from helpers.helpers import save_model

warnings.filterwarnings("ignore")


def simple_time_series(full_df, test_period, display_graphs=False):
    # This method are meant for displaying some info about this series in the main jupyter file
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
        print(
            "The mean absolute error (MAE) for the Simple Time Series model is {0:.2f}".format(
                find_RMSE(test, predictions)
            )
        )


def sts_predict_canteen_values(full_df, prediction_df, future=True):
    # Returns the predicted values for the prediction_df
    df = full_df.copy()
    df.index = pd.to_datetime(df.pop("date"))
    df = df.filter(["Canteen"])
    test_period = prediction_df.shape[0]

    if future is True:
        train = df
        test = prediction_df.filter(["Canteen"])
    else:
        train = df.iloc[:-test_period]
        test = df.iloc[-test_period:]

    _, predictions = prediction(train, test)
    return predictions


def prediction(train, test):
    # Binning the data
    bins = [-1, 1, 200, 500, 1000, 1500, 1700, 1800, 1900, 10000]
    binned_series, bin_means = bin_data(train, bins)

    train_x, train_y = get_lagged_list(binned_series, test.shape[0])

    # Create model
    model = create_sts_model(train_x, train_y)
    resulting_prediction = find_training_prediction(
        train_x, train_y, model, bin_means
    )
    predictions, pred_class = find_prediction_forecast(
        test, train_x, train_y, model, bin_means
    )

    return resulting_prediction, predictions


def bin_data(dataset, bins):
    binned = np.digitize(dataset.iloc[:, 0], bins)
    bin_means = {}

    for binn in range(1, len(bins)):
        bin_means[binn] = dataset[binned == binn].mean()

    return pd.Series(binned, index=dataset.index), bin_means


def get_lagged_list(binned_series, lags):
    lagged_list = []
    for s in range(lags):
        lagged_list.append(binned_series.shift(s))

    lagged_frame = pd.concat(lagged_list, 1).dropna()

    train_x = lagged_frame.iloc[:, 1:]
    train_y = lagged_frame.iloc[:, 0]
    return train_x, train_y


def get_mean_from_class(prediction, bin_means):
    return bin_means[prediction[0]]


def create_sts_model(train_x, train_y):
    model = GaussianNB()
    model.fit(train_x, train_y)
    save_model(model, "simple_time_series")
    return model


def find_training_prediction(train_x, train_y, model, bin_means):
    # Returns the prediction for the training set
    pred_insample = model.predict(train_x)
    pred_insample = pd.DataFrame(pred_insample, index=train_y.index)

    resulting_prediction = pd.Series(np.nan, index=train_y.index)
    for row in range(len(pred_insample)):
        mean_class = get_mean_from_class(pred_insample.values[row], bin_means)
        resulting_prediction.iloc[row] = mean_class[0]

    return resulting_prediction


def find_prediction_forecast(test, train_x, train_y, model, bin_means):
    # Returns the prediction for a test set (out of sample forecast), both predicted numbers and classes
    prediction_frame = pd.DataFrame(
        np.nan, index=test.index, columns=range(train_x.shape[1])
    )
    predictions = pd.Series(index=test.index)
    pred_class = pd.Series(index=test.index)

    prediction_frame.iloc[0, 1:] = train_x.iloc[-1, :-1].values
    prediction_frame.iloc[0, 0] = train_y.iloc[-1]

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


def test_stationarity(dataframe, time_window):
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


def find_RMSE(dataset, prediction):
    return np.sqrt(np.mean((dataset.iloc[:, 0] - prediction) ** 2))


def main():
    dt_df = pd.read_csv("../../data/decision_tree_df.csv")
    dt_df_test = dt_df.iloc[-8:]
    dt_df_test.index = pd.to_datetime(dt_df_test.pop("date"))

    canteen_prediction = sts_predict_canteen_values(dt_df_test)
    print(canteen_prediction)


if __name__ == "__main__":
    main()
