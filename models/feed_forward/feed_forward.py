import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import talos as ta
import numpy as np
from helpers.helpers import (
    normalize_dataset,
    preprocess,
    split_dataframe,
    plot_history,
)
from constants import ROOT_DIR
import warnings

warnings.filterwarnings("ignore")


class PrintDot(keras.callbacks.Callback):
    # Display training progress by printing a single dot for each completed epoch
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".", end="")


p = {
    "first_neuron": [10, 40, 160, 640, 1280],
    "hidden_neuron": [10, 40, 160],
    # 'hidden_layers':[0,1,2,4],
    # 'batch_size': [1000,5000,10000],
    # 'optimizer': ['adam'],
    # 'kernel_initializer': ['uniform'], #'normal'
    # 'epochs': [50],
    "dropout": [0.0, 0.25, 0.5],
    "last_activation": ["sigmoid"],
}


def canteen_model_optimize_parameters(
    train_dataset, train_labels, x_val, y_val, params
):
    model = keras.Sequential(
        [
            layers.Dense(
                params["first_neuron"], input_dim=16, activation=tf.nn.relu
            ),
            layers.Dropout(params["dropout"]),
            layers.Dense(params["hidden_neuron"], activation=tf.nn.relu),
            layers.Dropout(params["dropout"]),
            layers.Dense(1),
        ]
    )
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=["mean_absolute_error", "mean_squared_error"],
    )

    # early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    EPOCHS = 1000
    out = model.fit(
        train_dataset,
        train_labels,
        validation_data=[x_val, y_val],
        epochs=EPOCHS,
        verbose=0,
        # callbacks=[early_stop, PrintDot()],
    )
    return out, model


def calculate_optimized_parameters(train_dataset, train_labels):
    ta.Scan(
        x=(np.array(train_dataset)),
        y=(np.array(train_labels)),
        model=canteen_model_optimize_parameters,
        params=p,
        grid_downsample=0.50,
        dataset_name="canteen",
        experiment_no="3",
    )


def canteen_model(train_dataset, train_labels):
    model = keras.Sequential(
        [
            layers.Dense(
                64, input_dim=train_dataset.shape[1], activation=tf.nn.relu
            ),
            layers.Dense(512, activation=tf.nn.relu),
            layers.Dense(1),
        ]
    )
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=["mean_absolute_error", "mean_squared_error"],
    )

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
    EPOCHS = 1000
    history = model.fit(
        train_dataset,
        train_labels,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop],
    )
    return history, model


def plot_training_set(train_dataset):
    sns.pairplot(
        train_dataset[
            [
                "Canteen",
                "precipitation",
                "holiday",
                "vacation",
                "inneklemt",
                "canteen_week_ago",
                "canteen_day_ago",
                "preferred_work_temp",
                "stay_home_temp",
                "Monday",
                "Tuesday",
                "Friday",
            ]
        ],
        diag_kind="kde",
    )


def predict_canteen_values(dataset, to_predict):
    ml_df = dataset.copy()
    ml_df.index = pd.to_datetime(ml_df.pop("date"))
    train_dataset, test_dataset, train_labels, test_labels = preprocess(ml_df)
    normed_train_data, normed_test_data = normalize_dataset(
        train_dataset, test_dataset
    )
    history, model = canteen_model(normed_train_data, train_labels)
    # plot_history(history)
    model.save("{}/models/saved_models/feed_forward_model.h5".format(ROOT_DIR))
    predict_df = to_predict.copy()
    _, normed_predict_df = normalize_dataset(train_dataset, predict_df)
    predict_df["predicted_value"] = model.predict(normed_predict_df)
    predict_df = predict_df.filter(["date", "predicted_value"])
    return predict_df


def main():
    ml_df = pd.read_csv("{}/data/ml_df.csv".format(ROOT_DIR), index_col="date")
    test_prediction = ml_df.drop(["Canteen"], axis=1)
    res = predict_canteen_values(ml_df, test_prediction)
    print(res)


if __name__ == "__main__":
    main()
