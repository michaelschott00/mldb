import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mldb.store import RunStore

# Initialize store and load the German dataset
store = RunStore.from_env()
df = store.load_artifact_by_query(name="german", tags=["dataset"])

# Preprocess and split dataset
target = "credit_risk"
features = df.drop(columns=[target, "uuid"])
categorical_columns = features.select_dtypes(include=["object", "str"]).columns
ids = df["uuid"]
X = pd.get_dummies(features, columns=categorical_columns)
y = df.loc[:, target] - 1  # XGBoost needs 0/1 labels.
ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(
    ids, X, y, test_size=0.2, random_state=0
)

# Initialize xgboost run
xgboost_hparams = {"eval_metric": "logloss"}
run_id = store.create_run(hparams=xgboost_hparams, tags=["xgboost", "model", "german"])

# Fit an xgboost classifier and predict on all observations from both splits
model = xgb.XGBClassifier(**xgboost_hparams)
model.fit(X_train, y_train)

# Storing the observation uuid allows joining the full dataset later
for split, ids_split, X_split, y_split in zip(
    ["train", "test"], [ids_train, ids_test], [X_train, X_test], [y_train, y_test]
):
    predictions = pd.DataFrame(
        {
            "uuid": ids_split,
            "y_true": y_split,
            "predictions": model.predict(X_split),
        }
    )
    store.store(run_id, {f"predictions_{split}": predictions})

# Initialize logistic regression run
logreg_hparams = {"l1_ratio": 0.5, "solver": "saga"}
run_id = store.create_run(hparams=logreg_hparams, tags=["logreg", "model", "german"])

# Fit an xgboost classifier and predict on all observations from both splits
model = LogisticRegression(**logreg_hparams)  # type: ignore
model.fit(X_train, y_train)

# Storing the observation uuid allows joining the full dataset later
for split, ids_split, X_split, y_split in zip(
    ["train", "test"], [ids_train, ids_test], [X_train, X_test], [y_train, y_test]
):
    predictions = pd.DataFrame(
        {
            "uuid": ids_split,
            "y_true": y_split,
            "predictions": model.predict(X_split),
        }
    )
    store.store(run_id, {f"predictions_{split}": predictions})
