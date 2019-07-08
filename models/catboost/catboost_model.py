import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from helpers.helpers import preprocess


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


def main():
    raw_data = pd.read_csv("../../data/decision_tree_df.csv", index_col="date")
    df = preprocess_to_catboost(raw_data)
    train_dataset, test_dataset, train_labels, test_labels = preprocess(df)

    train_dataset_combined = Pool(train_dataset, train_labels)
    eval_dataset = Pool(test_dataset, test_labels)

    model = CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=5, eval_metric="MAE"
    )
    model.fit(train_dataset_combined, eval_set=eval_dataset)

    print(model.get_best_iteration())
    print(model.get_best_score())

    test_predictions = model.predict(test_dataset).flatten()

    plot_result(test_labels, test_predictions)

    # Predict next days
    raw_data = pd.read_csv(
        "../../data/decision_tree_df_next_days.csv", index_col="date"
    )
    processed_data = preprocess_to_catboost(raw_data)
    test1 = model.predict(processed_data).flatten()
    print(test1)
    plt.plot(test1)
    plt.show()


if __name__ == "__main__":
    main()
