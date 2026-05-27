import pandas as pd
from sklearn.datasets import make_classification

from mldb.store import RunStore

store = RunStore.from_env()
run_id = store.load("dataset")
# X, y =
