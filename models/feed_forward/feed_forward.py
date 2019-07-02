import os
import sys
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../analysis"
)
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../helpers")
from combined_dataset import open_csv
from helpers import normalize_dataset, preprocess

print(tf.__version__)


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".", end="")


def build_model(train_dataset):
    model = keras.Sequential(
        [
            layers.Dense(
                64,
                activation=tf.nn.relu,
                input_shape=[len(train_dataset.keys())],
            ),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1),
        ]
    )
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=["mean_absolute_error", "mean_squared_error"],
    )
    return model


def plot_training_set(train_dataset):
    sns.pairplot(
        train_dataset[
            [
                "Canteen",
                "precipitation",
                "max_temp",
                "min_temp",
                "holiday",
                "vacation",
            ]
        ],
        diag_kind="kde",
    )


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


def fit_model(model, normed_train_data, train_labels):
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    EPOCHS = 1000
    history = model.fit(
        normed_train_data,
        train_labels,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop, PrintDot()],
    )
    return history


def main():
    train_dataset, test_dataset, train_labels, test_labels = preprocess(
        open_csv("../data/dataset.csv")
    )
    normed_train_data, normed_test_data = normalize_dataset(
        train_dataset, test_dataset
    )
    model = build_model(train_dataset)
    history = fit_model(model, normed_train_data, train_labels)
    plot_history(history)


if __name__ == "__main__":
    main()
