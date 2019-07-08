import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from statsmodels.tsa.stattools import adfuller


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


def create_model(train_x, train_y):
    model = GaussianNB()
    model.fit(train_x, train_y)
    return model


def find_training_prediction(train_x, train_y, model, bin_means):
    pred_insample = model.predict(train_x)
    pred_insample = pd.DataFrame(pred_insample, index=train_y.index)

    resulting_prediction = pd.Series(np.nan, index=train_y.index)
    for row in range(len(pred_insample)):
        mean_class = get_mean_from_class(pred_insample.values[row], bin_means)
        resulting_prediction.iloc[row] = mean_class[0]

    return resulting_prediction


def find_prediction_forecast(test, train_x, train_y, model, bin_means):
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
