# MLDB

[![MLDB](https://github.com/michaelschott00/mldb/actions/workflows/python-app.yml/badge.svg)](https://github.com/michaelschott00/mldb/actions/workflows/python-app.yml)

MLDB is a metadata-indexed blob store for machine learning experiments. It lets you store datasets, artifacts, and results without ever having to think about where they live on disk — you tag things with metadata (tags, hyperparameters, run info) instead of inventing file paths, and you get them back the same way.

The result: no more `model_x_lr=0.1_batch_size=512_cosine.csv`, no more digging through nested output folders to figure out which run produced which file.

## The best part: query your results straight into DuckDB

Once your runs and artifacts are stored with metadata, MLDB can assemble a DuckDB database containing exactly the tables you care about — selected by tag query — so you can analyze results with plain SQL instead of writing custom loading code.

```python
db = store.get_db()
db.attach(names=["predictions_test"], tags=["model", "german"])
db.attach(names=["german"])

with db.connect() as con:
    result = con.sql("""
        select
            personal_status_sex as "Personal status and sex",
            case when p.logreg then 'Logistic' else 'XGBoost' end as model,
            avg(case when y_true != predictions then 1 else 0 end) as accuracy
        from predictions_test p
        join german d on p.uuid = d.uuid
        group by p.logreg, personal_status_sex
    """).df()
```

No file paths, no manual joins across scattered CSVs — just the tables you asked for, ready to query.

See [`docs/examples/simple_example/example.ipynb`](docs/examples/simple_example/example.ipynb) for the full walkthrough, from downloading a dataset to comparing two models' predictions.

## How it works

- **Runs** are the unit of experiment tracking. Each run is created with a set of tags and hyperparameters, and gets a unique ID plus a friendly, memorable name.
- **Artifacts** (datasets, predictions, results, anything storable as a table or file) are attached to a run and are automatically indexed by that run's metadata.
- **Queries** let you find runs and artifacts by tags and hyperparameters instead of by path — `store.list_runs(tags=[...])`, `store.list_artifacts_by_query(tags=[...])`, `store.load_artifact_by_query(...)`.
- A **command-line interface** (`mldb list`, `mldb artifacts`, ...) gives you the same lookups without leaving the terminal.

## Installation

**Coming soon** — MLDB isn't published to PyPI yet. In the meantime, clone the repository and install it locally:

```bash
git clone https://github.com/michaelschott00/mldb.git
cd mldb
pip install -e .
```

## Getting started

The [`docs/examples`](docs/examples) directory contains runnable notebooks:

- [`simple_example`](docs/examples/simple_example) — store datasets and model predictions, then query and analyze them together with DuckDB.
- [`tuning_example (wip)`](docs/examples/tuning_example) — track and compare runs across a hyperparameter sweep.
- [`tensorboard_example`](docs/examples/tensorboard_example) — combine MLDB with TensorBoard logging.

## License

MIT — see [LICENCE.txt](LICENCE.txt).
