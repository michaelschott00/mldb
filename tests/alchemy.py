# SQLAlchemy

import sqlite3
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    and_,
    create_engine,
    delete,
    insert,
    or_,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import aliased
from sqlalchemy.schema import CreateTable

# Example data

tag_df = pd.DataFrame(
    {
        "run_id": [
            "run_1",
            "run_4",
            "run_5",
            "run_2",
            "run_2",
            "run_3",
            "run_3",
        ],
        "tag": [
            "german",
            "adult",
            "iris",
            "xgboost_baseline",
            "german",
            "xgboost_best",
            "german",
        ],
    }
)

artifact_df = pd.DataFrame(
    {
        "run_id": [
            "run_1",
            "run_4",
            "run_5",
            "run_2",
            "run_2",
            "run_2",
            "run_3",
            "run_3",
            "run_3",
        ],
        "artifact_name": [
            "dataset",
            "dataset",
            "dataset",
            "predictions_train",
            "predictions_val",
            "predictions_test",
            "predictions_train",
            "predictions_val",
            "predictions_test",
        ],
        "checksum": [
            "aaa",
            "bbb",
            "ccc",
            "ddd",
            "eee",
            "fff",
            "ggg",
            "hhh",
            "iii",
        ],
    }
)

hparam_df = pd.DataFrame(
    {
        "run_id": [
            "run_2",
            "run_2",
            "run_3",
            "run_3",
        ],
        "hparam": ["eval_logs", "lr", "eval_logs", "lr"],
        "value": ["log_loss", "1e-3", "squared_loss", "1e-4"],
    }
)

# SQL alchemy setup
_engine: Engine = create_engine(f"sqlite:///:memory:")

tag_df.to_sql("tags", _engine, if_exists="replace")
artifact_df.to_sql("artifacts", _engine, if_exists="replace")
hparam_df.to_sql("hparams", _engine, if_exists="replace")

_meta = MetaData()
_artifacts = Table(
    "artifacts",
    _meta,
    Column("run_id", String, primary_key=True),
    Column("artifact_name", String, primary_key=True),
    Column("artifact_checksum", String, nullable=False),
)
_hparams = Table(
    "hparams",
    _meta,
    Column("run_id", String, primary_key=True),
    Column("hparam", String, nullable=False),
    Column("value", String, nullable=False),
)
_tags = Table(
    "tags",
    _meta,
    Column("run_id", String, primary_key=True),
    Column("tag", String, primary_key=True),
)

# Spec


@dataclass
class spec:
    tables: list[str]
    hparams: dict[str, list[str]] | None = None
    collections: list[str] | None = None


specs = (
    spec(
        tables=["predictions_train", "predictions_val", "predictions_test"],
        hparams={"eval_logs": ["logloss"], "lr": ["1e-3", "1e-2"]},
        collections=["xgboost_baseline", "xgboost_best"],
    ),
    spec(tables=["dataset"], collections=["german"]),
)

hparams = []
for entry in specs:
    if entry.hparams is not None:
        hparams.extend(list(entry.hparams.keys()))

# SQL Query

aliases = [aliased(_hparams) for _ in range(len(hparams))]
h1 = aliased(_hparams)
h2 = aliased(_hparams)
cte = (
    select(h1.c.run_id, h1.c.value.label("eval_logs"), h2.c.value.label("lr"))
    .join_from(
        h1,
        h2,
        and_(
            h1.c.run_id == h2.c.run_id,
            h1.c.hparam == "eval_logs",
            h2.c.hparam == "lr",
        ),
    )
    .cte()
)
stmt = (
    select(
        _artifacts.c.run_id,
        _artifacts.c.artifact_name,
        _tags.c.tag,
        cte.c.eval_logs,
        cte.c.lr,
    )
    .join(_tags, _tags.c.run_id == _artifacts.c.run_id)
    .outerjoin(cte, _tags.c.run_id == cte.c.run_id)
    .where(
        or_(
            and_(
                _tags.c.tag.in_(["xgboost_baseline", "xgboost_best"]),
                cte.c.eval_logs == "log_loss",
                cte.c.lr == "1e-3",
                _artifacts.c.artifact_name.in_(
                    ["predictions_train", "predictions_val", "predictions_test"]
                ),
            ),
            and_(_artifacts.c.artifact_name == "dataset", _tags.c.tag == "german"),
        )
    )
)
with _engine.connect() as conn:
    for table in [_artifacts, _hparams, _tags]:
        conn.execute(CreateTable(table, if_not_exists=True))
    conn.commit()
    res = conn.execute(stmt).all()
print(pd.DataFrame(res))

# Cleanup

_engine.dispose()
