import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from helpers.helpers import preprocess, save_model, load_model, plot_history_df
import os
from constants import ROOT_DIR


def preprocess_to_catboost(raw_data):
    df = raw_data.copy()
    df["weekday"] = df["weekday"].map(
        {
            "Monday": 1,
            "Tuesday": 2,
            "Wednesday": 3,
            "Thursday": 4,
            "Friday": 5,
            "Saturday": 6,
            "Sunday": 7,
        }
    )
    df["avg_temp"] = df["avg_temp"].map(
        {"stay_home_temp": 1, "preferred_work_temp": 2}
    )
    return df


def plot_result(test_labels, test_predictions):
    plt.scatter(
        list(test_labels["Canteen"]),
        test_predictions,
        c="b",
        label="prediction",
    )
    plt.xlabel("True Values [Canteen]")
    plt.ylabel("Predictions [Canteen]")
    plt.axis("equal")
    plt.axis("square")
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([-10000, 10000], [-10000, 10000], c="r", label="actual")
    plt.legend(loc="best")
    plt.show()

    error = test_predictions - list(test_labels["Canteen"])
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [Canteen]")
    plt.ylabel("Count")
    plt.show()


def catboost_predict_values(dt_df, df_to_predict):
    if os.path.isfile("{}/models/saved_models/catboost.sav".format(ROOT_DIR)):
        model = load_model("catboost")
    else:
        model = catboost_create_model(dt_df)

    # Predict input
    df = df_to_predict.copy()
    prepros_df = preprocess_to_catboost(df)
    prediction = model.predict(prepros_df)
    df["predicted_value"] = prediction
    return df.filter(["date", "predicted_value"])


def catboost_create_model(dt_df):
    raw_data = dt_df.copy()
    raw_data.index = pd.to_datetime(raw_data.pop("date"))
    df = preprocess_to_catboost(raw_data)
    train_dataset, test_dataset, train_labels, test_labels = preprocess(df)

    train_dataset_combined = Pool(train_dataset, train_labels)
    eval_dataset = Pool(test_dataset, test_labels)

    model = CatBoostRegressor(
        iterations=2000, learning_rate=0.05, depth=5, eval_metric="MAE"
    )
    model.fit(train_dataset_combined, eval_set=eval_dataset, verbose=0)
    save_model(model, "catboost")
    return model


def main():
    dt_df = pd.read_csv("../../data/decision_tree_df.csv")
    dt_df_test = dt_df.iloc[-8:]
    dt_df_test.index = pd.to_datetime(dt_df_test.pop("date"))
    dt_df_test = dt_df_test.drop(["Canteen"], axis=1)
    model = catboost_create_model(dt_df)
    print(catboost_predict_values(dt_df, dt_df_test))
    plot_history_df(model.get_evals_result())


if __name__ == "__main__":
    main()
