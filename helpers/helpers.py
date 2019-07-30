import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from constants import ROOT_DIR


def all_paths_in_dir(path, file_type=".txt"):
    files = []
    for r, d, f in os.walk(path):
        for file_path in f:
            if file_type in file_path:
                files.append(os.path.join(r, file_path))
    if not files:
        print("No {} files in {}, is it correct path?".format(file_type, path))
    return files


def load_excel_files_into_df(file_path, file_type):
    parking_data = all_paths_in_dir(file_path, file_type)
    df_list = []
    for elem in parking_data:
        df_list.append(pd.read_excel(elem))
    return pd.concat(df_list)


def split_dataframe(df, split_elements_list):
    """
    Splits a dataframe and returns two new dfs with the split_elements in the new df
    :param df: dataframe to be splitted
    :param split_elements_list: array of elements to remove from old df (x) and place into new (y)
    :return: x (old df) and y (new df)
    """
    y = df.filter(split_elements_list)
    x = df.drop(split_elements_list, axis=1)

    return x, y


def map_bool_to_int(df, col):
    df[col] = df[col].map({True: 1.0, False: 0.0})


def normalize_dataset(train_dataset, test_dataset):
    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()

    def norm(x):
        return (x - train_stats["mean"]) / train_stats["std"]

    return norm(train_dataset), norm(test_dataset)


def save_model(model, filename):
    filename = "{}/models/saved_models/{}.sav".format(ROOT_DIR, filename)
    pickle.dump(model, open(filename, "wb"))


def load_model(filename):
    filename = "{}/models/saved_models/{}.sav".format(ROOT_DIR, filename)
    loaded_model = pickle.load(open(filename, "rb"))
    return loaded_model


def preprocess(raw_dataset):
    df = raw_dataset.copy()
    train, test = train_test_split(df, test_size=0.2)
    train_dataset, train_labels = split_dataframe(train, ["Canteen"])
    test_dataset, test_labels = split_dataframe(test, ["Canteen"])
    return train_dataset, test_dataset, train_labels, test_labels


def plot_history(history, epoch):
    hist = pd.DataFrame(history)
    hist["epoch"] = epoch
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error [MPG]")
    plt.plot(hist["epoch"], hist["mean_absolute_error"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mean_absolute_error"], label="Val Error")
    plt.legend()


def plot_history_df(model):
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error [MPG]")
    plt.plot(model["learn"]["MAE"], label="Train Error")
    plt.plot(model["validation"]["MAE"], label="Val Error")
    # plt.ylim([0,5])
    plt.legend()


def plot_prediction(test_prediction):
    plt.scatter(
        list(test_prediction["Canteen"]),
        list(test_prediction["prediction"]),
        c="b",
        label="Prediction",
    )
    plt.xlabel("True Values [Canteen]")
    plt.ylabel("Predictions [Canteen]")
    plt.axis("equal")
    plt.axis("square")
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([-10000, 10000], [-10000, 10000], c="r", label="Actual")
    plt.legend(loc="best")


def plot_history_and_prediction_df(model, test_prediction):
    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    plot_history_df(model)
    plt.subplot(122)
    plot_prediction(test_prediction)
    plt.show()


def plot_history_and_prediction_ml(history, epoch, test_prediction):
    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    plot_history(history, epoch)
    plt.subplot(122)
    plot_prediction(test_prediction)
    plt.show()
