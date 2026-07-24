import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from mldb.store import RunStore

store = RunStore.from_env()
df = store.load_by_tags_single("german", ["dataset"]).artifact

target = "credit_risk"
categorical_columns = (
    df.drop(columns=[target]).select_dtypes(include=["object", "str"]).columns
)
X = pd.get_dummies(df.drop(columns=[target]), columns=categorical_columns)
# German dataset encodes risk as 1 (good) / 2 (bad); XGBoost needs 0/1 labels.
y = df.loc[:, target] - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = xgb.XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

predictions = pd.DataFrame(
    {
        "y_true": y_test.reset_index(drop=True),  # type: ignore
        "y_pred": model.predict(X_test),
    }
)

run_id = store.create_run(tags=["example", "german", "model"])
store.store(run_id, {"predictions": predictions})
