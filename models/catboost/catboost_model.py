import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from helpers.helpers import preprocess

raw_data = pd.read_csv("../../data/decision_tree_df.csv", index_col="date")
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

train_dataset, test_dataset, train_labels, test_labels = preprocess(df)

model = CatBoostRegressor()
parameters = {
    "depth": [6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "iterations": [30, 50, 100],
}
grid = GridSearchCV(estimator=model, param_grid=parameters, cv=4, n_jobs=-1)
grid.fit(train_dataset, train_labels)

print("The best parameters across ALL searched params:\n", grid.best_params_)

preds = grid.predict(test_dataset)

plt.figure(figsize=(20, 10))
plt.plot(preds, color="r")
plt.plot(test_labels, color="b")
plt.show()
