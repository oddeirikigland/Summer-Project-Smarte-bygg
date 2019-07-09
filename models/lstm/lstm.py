import os
import sys
from numpy import concatenate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.python.keras import Sequential, optimizers
from tensorflow.python.keras.layers import LSTM, Dense, Dropout

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../helpers")
from helpers import (
    normalize_dataset,
    preprocess,
    map_bool_to_int,
    split_dataframe,
)


def build_model(
    train_dataset, train_labels, test_dataset, test_labels, local_testing
):
    model = Sequential()
    model.add(
        LSTM(5, input_shape=(train_dataset.shape[1], train_dataset.shape[2]))
    )
    model.add(Dense(1))
    optimizer = optimizers.Adam(lr=0.01)
    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=["mean_absolute_error", "mean_squared_error"],
    )
    if local_testing:
        history = model.fit(
            train_dataset,
            train_labels,
            epochs=200,
            batch_size=20,
            validation_data=(test_dataset, test_labels),
            verbose=0,
            shuffle=False,
        )
    else:
        history = model.fit(
            train_dataset,
            train_labels,
            epochs=200,
            batch_size=20,
            verbose=0,
            shuffle=False,
        )

    return model, history


def reshape_df(np_array):
    np_array = np.asarray(np_array)
    np_array = np_array.reshape((np_array.shape[0], 1, np_array.shape[1]))
    return np_array


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error [MPG]")
    plt.plot(hist["epoch"], hist["mean_absolute_error"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mean_absolute_error"], label="Val Error")
    # plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error [$MPG^2$]")
    plt.plot(hist["epoch"], hist["mean_squared_error"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mean_squared_error"], label="Val Error")
    # plt.ylim([0,20])
    plt.legend()
    plt.show()


def create_train_dataset(supervised_scaled, test_period):
    train = supervised_scaled.iloc[:-test_period]
    test = supervised_scaled.iloc[-test_period:]

    train_dataset, train_labels = split_dataframe(train, ["Canteen"])
    test_dataset, test_labels = split_dataframe(test, ["Canteen"])

    train_dataset = reshape_df(train_dataset)
    test_dataset = reshape_df(test_dataset)  # reshape_df(test_dataset)
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    return train_dataset, test_dataset, train_labels, test_labels


def lstm(df, test_period, local_testing=False):
    df.fillna(0, inplace=True)
    df.dropna(inplace=True)
    df.inneklemt = df.inneklemt.astype(float)

    columns = [
        "Canteen",
        "precipitation",
        "holiday",
        "vacation",
        "inneklemt",
        "canteen_week_ago",
        "canteen_day_ago",
        "dist_start_year",
        "preferred_work_temp",
        "stay_home_temp",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    supervised = df.copy()
    supervised = supervised[columns]

    values = supervised.astype("float64")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    supervised_scaled = pd.DataFrame(scaled)
    supervised_scaled.columns = columns

    train_dataset, test_dataset, train_labels, test_labels = create_train_dataset(
        supervised_scaled, test_period
    )

    model, history = build_model(
        train_dataset, train_labels, test_dataset, test_labels, local_testing
    )

    return model, scaler, test_dataset, test_labels, history


# Use when predicting with existing data in ml_df.csv
def predict_lstm_with_testset(period):
    df = pd.read_csv("../../data/ml_df.csv", index_col="date")
    model, scaler, test_dataset, test_labels, history = lstm(
        df, period, local_testing=True
    )

    # From tutorial https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    yhat = model.predict(test_dataset)
    test_dataset = test_dataset.reshape(
        (test_dataset.shape[0], test_dataset.shape[2])
    )
    inv_yhat = concatenate((yhat, test_dataset), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    test_y = test_labels.reshape((len(test_labels), 1))

    inv_y = concatenate((test_y, test_dataset), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # calculate Mean Squared error
    mae = mean_absolute_error(inv_y, inv_yhat)
    print("Test Mean Squared Absolute Error: %.3f" % mae)
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print("Test RMSE: %.3f" % rmse)
    return history, inv_yhat


# Use when predicting for future with dataset that is NOT in ml_df.csv
def predict_future_with_real_data(test_df):
    df = pd.read_csv("../../data/ml_df.csv", index_col="date")

    if "Canteen" not in test_df.columns:
        test_df["Canteen"] = np.nan

    df = df.append(test_df, sort=False)

    model, scaler, test_dataset, test_labels, history = lstm(
        df, test_df.shape[0]
    )

    yhat = model.predict(test_dataset)
    test_dataset = test_dataset.reshape(
        (test_dataset.shape[0], test_dataset.shape[2])
    )
    inv_yhat = concatenate((yhat, test_dataset), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    return inv_yhat


def main():
    # df = pd.read_csv("../../data/test_data.csv", index_col="date")
    # predict_future_with_real_data(df)
    history, _ = predict_lstm_with_testset(8)
    plot_history(history)


if __name__ == "__main__":
    main()
