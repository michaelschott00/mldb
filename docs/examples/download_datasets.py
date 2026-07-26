import io
import uuid

import pandas as pd
import requests

from mldb.store import RunStore

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
GERMAN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

GERMAN_COLUMNS = [
    "status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "credit_risk",
]


def _add_uuid_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a uuid column identifying each row, so results can be joined back to it without storing the full dataset again."""
    df = df.copy()
    df.insert(0, "uuid", [str(uuid.uuid4()) for _ in range(len(df))])
    return df


def download_adult() -> pd.DataFrame:
    """Download and parse the UCI Adult dataset."""
    response = requests.get(ADULT_URL)
    response.raise_for_status()
    df = pd.read_csv(
        io.StringIO(response.text),
        names=ADULT_COLUMNS,
        sep=",",
        skipinitialspace=True,
        na_values="?",
    ).dropna()
    return _add_uuid_column(df)


def download_german() -> pd.DataFrame:
    """Download and parse the UCI Statlog German Credit dataset."""
    response = requests.get(GERMAN_URL)
    response.raise_for_status()
    df = pd.read_csv(
        io.StringIO(response.text),
        names=GERMAN_COLUMNS,
        sep=r"\s+",
    )
    return _add_uuid_column(df)


def main() -> None:
    store = RunStore("./data")
    try:
        run_id = store.create_run(tags=["dataset"])
        store.store(
            run_id,
            {
                "adult": download_adult(),
                "german": download_german(),
            },
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
