import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime as dt
from sklearn.metrics import mean_absolute_error, r2_score
from helpers.helpers import save_model, is_model_saved, load_model
from constants import ROOT_DIR, DATA_SET_TEST_SIZE


def linear_create_model(df):
    dataframe = pd.DataFrame(df.index)
    dataframe["Canteen"] = df["Canteen"].values
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe["date"] = dataframe["date"].map(dt.datetime.toordinal)

    train, test = train_test_split(dataframe, test_size=DATA_SET_TEST_SIZE)

    y = np.asarray(train["Canteen"])
    x = np.asarray(train["date"]).reshape((-1, 1))
    model = LinearRegression(normalize=True).fit(
        x, y
    )  # create linear regression object #train model on train data
    save_model(model, "linear_regression")
    return model, x, y, test


def linear(df):
    model, x, y, test = linear_create_model(df)
    # Changing ordinal dates to regular dates
    x = [dt.datetime.fromordinal(*el).date() for el in x]

    y_test = np.asarray(test["Canteen"])
    x_test = np.asarray(test["date"]).reshape((-1, 1))

    y_pred = model.predict(x_test)

    print("Mean absolute error: %.0f" % mean_absolute_error(y_test, y_pred))

    return x, y, y_pred, x_test, y_test


def main():
    dataset = pd.read_csv(
        "{}/data/dataset.csv".format(ROOT_DIR), index_col="date"
    )
    linear(dataset)


if __name__ == "__main__":
    main()
