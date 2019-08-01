from numpy import concatenate
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.python.keras import Sequential, optimizers
from tensorflow.python.keras.layers import LSTM, Dense
from constants import ROOT_DIR, DATA_SET_TEST_SIZE
from helpers.helpers import split_dataframe, plot_history, save_model
import os


def build_model(train_dataset, train_labels, local_testing):
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
            validation_split=0.2,
            verbose=2,
            shuffle=False,
        )

    else:
        history = model.fit(
            train_dataset,
            train_labels,
            epochs=200,
            batch_size=20,
            validation_split=0.2,
            verbose=0,
            shuffle=False,
        )

    return model, history


def reshape_df(np_array):
    np_array = np.asarray(np_array)
    np_array = np_array.reshape((np_array.shape[0], 1, np_array.shape[1]))
    return np_array


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


def preprocess_data_for_model(df):
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
    return scaler, supervised_scaled


# Train a new model
def train_lstm(df, test_period, local_testing=False):
    scaler, supervised_scaled = preprocess_data_for_model(df)

    train_dataset, test_dataset, train_labels, test_labels = create_train_dataset(
        supervised_scaled, test_period
    )

    model, history = build_model(train_dataset, train_labels, local_testing)

    return model, scaler, test_dataset, test_labels, history


# Option for using an existing model with MAE score less than 75.
def load_existing_lstm(df, test_period):
    scaler, supervised_scaled = preprocess_data_for_model(df)

    _, test_dataset, _, _ = create_train_dataset(
        supervised_scaled, test_period
    )
    model = load_model("{}/models/saved_models/lstm_model.h5".format(ROOT_DIR))

    return model, scaler, test_dataset


# Use when predicting with existing data in ml_df.csv, stores model if MAE is less than 75.
def predict_lstm_with_testset(ml_df, period, local_testing=True):
    df = ml_df.copy()
    model, scaler, test_dataset, test_labels, history = train_lstm(
        df, period, local_testing
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
    # mae = mean_absolute_error(inv_y, inv_yhat)
    # print("Test Mean Absolute Error: %.3f" % mae)
    model.save("{}/models/saved_models/lstm_model.h5".format(ROOT_DIR))
    save_model(history.history, "lstm_history")
    save_model(history.epoch, "lstm_epoch")
    save_model(inv_yhat, "lstm_prediction")
    return history, inv_yhat


# Use when predicting for future with dataset that is NOT in ml_df.csv
def predict_future_with_real_data(ml_df, t_df):
    df = ml_df.copy()
    test_df = t_df.copy()

    if "Canteen" not in test_df.columns:
        test_df["Canteen"] = np.nan

    df = df.append(test_df, sort=False)

    model, scaler, test_dataset, test_labels, history = train_lstm(
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


# Assumes canteen data is not given
def predict_future_with_trained_model_file(ml_df, dataset):
    test_dataset = dataset.copy()
    df = ml_df.copy()

    if "Canteen" not in test_dataset.columns:
        test_dataset["Canteen"] = np.nan

    df = df.append(test_dataset, sort=False)

    if not os.path.isfile(
        "{}/models/saved_models/lstm_model.h5".format(ROOT_DIR)
    ):
        return predict_lstm_with_testset(ml_df, test_dataset.shape[0])[1]

    model, scaler, test_dataset = load_existing_lstm(df, test_dataset.shape[0])
    yhat = model.predict(test_dataset)
    test_dataset = test_dataset.reshape(
        (test_dataset.shape[0], test_dataset.shape[2])
    )
    inv_yhat = concatenate((yhat, test_dataset), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    return inv_yhat


def lstm_create_model(ml_df):
    predict_lstm_with_testset(
        ml_df, int(ml_df.shape[0] * DATA_SET_TEST_SIZE), local_testing=False
    )


def main():
    df = pd.read_csv("{}/data/ml_df.csv".format(ROOT_DIR), index_col="date")
    df.drop(df.tail(8).index, inplace=True)
    hist, inv = predict_lstm_with_testset(df, 172)
    # plot_history(hist)

    # predict_future_with_real_data(df)
    # predict_future_with_trained_model_file(df)
    # predict_lstm_with_testset(8)


if __name__ == "__main__":
    main()
