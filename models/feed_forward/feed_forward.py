import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from helpers.helpers import (
    normalize_dataset,
    preprocess,
    save_model,
    plot_history,
    load_model_sav,
    is_model_saved,
)
from constants import ROOT_DIR
import warnings
import numpy as np

warnings.filterwarnings("ignore")


class PrintDot(keras.callbacks.Callback):
    # Display training progress by printing a single dot for each completed epoch
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".", end="")


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


def feed_forward_create_model(ml_df):
    ml_df = ml_df.copy()
    train_dataset, test_dataset, train_labels, test_labels = preprocess(ml_df)
    normed_train_data, normed_test_data = normalize_dataset(
        train_dataset, test_dataset
    )
    history, model = canteen_model(normed_train_data, train_labels)

    test_labels["prediction"] = model.predict(normed_test_data).flatten()

    print(
        mean_absolute_error(
            np.asarray(test_labels["Canteen"]),
            np.asarray(test_labels["prediction"]),
        )
    )

    save_model(test_labels, "feed_forward_test_set_prediction")

    model.save("{}/models/saved_models/feed_forward_model.h5".format(ROOT_DIR))
    save_model(history.history, "feed_forward_history")
    save_model(history.epoch, "feed_forward_epoch")
    save_model(train_dataset, "feed_forward_train_dataset")
    return model


def predict_canteen_values(dataset, to_predict):
    ml_df = dataset.copy()
    if is_model_saved("feed_forward_model.h5"):
        model = keras.models.load_model(
            "{}/models/saved_models/feed_forward_model.h5".format(ROOT_DIR)
        )
    else:
        model = feed_forward_create_model(ml_df)
    predict_df = to_predict.copy()
    _, normed_predict_df = normalize_dataset(
        load_model_sav("feed_forward_train_dataset"), predict_df
    )
    predict_df["predicted_value"] = model.predict(normed_predict_df)
    predict_df = predict_df.filter(["date", "predicted_value"])
    return predict_df


def main():
    ml_df = pd.read_csv("{}/data/ml_df.csv".format(ROOT_DIR))
    test_prediction = ml_df.drop(["Canteen"], axis=1)
    test_prediction.index = pd.to_datetime(test_prediction.pop("date"))
    res = predict_canteen_values(ml_df, test_prediction)
    print(res.head())

    plot_history(
        load_model_sav("feed_forward_history"),
        load_model_sav("feed_forward_epoch"),
    )


if __name__ == "__main__":
    main()
