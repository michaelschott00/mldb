import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from mldb.store import RunStore

# Initialize store and load the German dataset
store = RunStore.from_env()
df = store.load_artifact_by_tags("german", ["dataset"]).artifact

# Preprocess and split dataset
target = "credit_risk"
features = df.drop(columns=[target, "uuid"])
categorical_columns = features.select_dtypes(include=["object", "str"]).columns
X = pd.get_dummies(features, columns=categorical_columns)

# German dataset encodes risk as 1 (good) / 2 (bad); XGBoost needs 0/1 labels.
y = df.loc[:, target] - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit an xgboost classifier and predict on all observations from both splits
model = xgb.XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

predictions = pd.DataFrame(
    {
        "uuid": df["uuid"],
        "y_true": y,
        "predictions": model.predict(X),
    }
)

# Store only the uuid, true label and prediction; join on uuid to recover the rest of the dataset
run_id = store.create_run(tags=["example_run_1", "german", "model"])
store.store(run_id, {"predictions": predictions})
