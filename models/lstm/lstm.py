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
import warnings

warnings.filterwarnings("ignore")
LSTM_MAE = 1000


def build_model(train_dataset, train_labels, local_testing):
    """
    Function for building the LSTM model and setting hyperparameters
    :param train_dataset: numpy array with all necessary variables, does not contain canteen values
    :param train_labels: the canteen values that is the solution/real values.
    :param local_testing: bool value, will print training data if set to True.
    :return: The model that is built and the training history
    """
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


def reshape_df(df):
    """
    Takes a dataframe and converts it to a numpy array with a certain shape
    :param df: dataframe
    :return: a numpy array with shape (?,1,?)
    """
    np_array = np.asarray(df)
    np_array = np_array.reshape((np_array.shape[0], 1, np_array.shape[1]))
    return np_array


def create_train_dataset(supervised_scaled, test_period):
    """
    Splits a dataframe into test and train sets
    :param supervised_scaled: a dataframe
    :param test_period: int for setting the test size
    :return: train_dataset, test_dataset, train_labels, test_labels
    """
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
    """
    Preprocesses a dataframe by scaling the dataframe to be in the range of 0 to 1
    :param df: a full dataframe
    :return: scaler for scaling the dataframe back to real values and the scaled supervised dataframe
    """
    df.fillna(0, inplace=True)
    df.dropna(inplace=True)
    df.inneklemt = df.inneklemt.astype(float)

    columns = [
        "Canteen",
        "holiday",
        "vacation",
        "inneklemt",
        "canteen_week_ago",
        "canteen_day_ago",
        "dist_start_year",
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


def load_existing_lstm(df, test_period):
    """
    Loads an existing model from a .h5 file
    :param df: a full dataframe
    :param test_period: int for setting the test size
    :return: trained model from file, scaler and test dataset
    """
    scaler, supervised_scaled = preprocess_data_for_model(df)

    _, test_dataset, _, _ = create_train_dataset(
        supervised_scaled, test_period
    )
    model = load_model("{}/models/saved_models/lstm_model.h5".format(ROOT_DIR))

    return model, scaler, test_dataset


def predict_lstm_with_testset(ml_df, test_period, local_testing=True):
    """
    LSTM prediction with existing data in ml_df.csv for training the model
    :param ml_df: a dataframe that is non categorical
    :param test_period: int for setting the test size
    :param local_testing: bool value, will print training data if set to True.
    :return: model history and the predicted values
    """
    df = ml_df.copy()
    model, scaler, test_dataset, test_labels, history = train_lstm(
        df, test_period, local_testing
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

    pred_df = pd.DataFrame(
        {"prediction": inv_yhat.flatten(), "Canteen": inv_y.flatten()}
    )
    mae = mean_absolute_error(
        np.asarray(pred_df["Canteen"]), np.asarray(pred_df["prediction"])
    )
    global LSTM_MAE
    if mae < LSTM_MAE:
        model.save("{}/models/saved_models/lstm_model.h5".format(ROOT_DIR))
        save_model(history.history, "lstm_history")
        save_model(history.epoch, "lstm_epoch")
        save_model(pred_df, "lstm_test_set_prediction")
        LSTM_MAE = mae
        print("LSTM", str(mae))
    return history, inv_yhat


# Use when predicting for future with dataset that is NOT in ml_df.csv
def predict_future_with_real_data(ml_df, t_df):
    """
    This method should be used for predicting the future (next 8 days).
    :param ml_df: a dataframe for training the model on.
    :param t_df: a dataframe with data for the next 8 days filled out. t_df should not contain the same data as ml_df.
    :return: the predicted values of the model.
    """
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
def predict_future_with_trained_model_file(ml_df, t_df):
    """
    Used for making predictions on a trained model file
    :param ml_df: a full dataframe
    :param t_df: a dataframe with data for the next 8 days filled out. t_df should not contain the same data as ml_df.
    :return: the predicted values of the model.
    """
    test_dataset = t_df.copy()
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
