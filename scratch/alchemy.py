# SQLAlchemy

import pandas as pd
from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    and_,
    create_engine,
    or_,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import aliased
from sqlalchemy.schema import CreateTable

# Example data

runs_df = pd.DataFrame(
    {
        "run_id": [
            "run_1",
            "run_2",
            "run_3",
            "run_4",
            "run_5",
        ],
        "run_name": [
            "flunky flipper",
            "floppy flupper",
            "flippy flapper",
            "babbl bragger",
            "snipper snapper",
        ],
    },
)

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
runs_df.to_sql("runs", _engine, if_exists="replace")

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
_runs = Table(
    "runs",
    _meta,
    Column("run_id", String, primary_key=True),
    Column("run_name", String, primary_key=False),
)

with _engine.connect() as conn:
    for table in [_artifacts, _hparams, _tags, _runs]:
        conn.execute(CreateTable(table, if_not_exists=True))
    conn.commit()

# Experiments


def flatten_hparams():
    # names=["predictions_train", "predictions_val", "predictions_test"],
    hparam_spec = {"eval_logs": ["log_loss"], "lr": ["1e-3"]}
    # collections=["xgboost_baseline", "xgboost_best"],

    # SQL Query

    hparam_names = list(hparam_spec.keys())
    hparam_tables = [aliased(_hparams) for _ in range(len(hparam_names))]
    hparams_stmt = select(
        hparam_tables[0].c.run_id,
        *[t.c.value.label(n) for t, n in zip(hparam_tables, hparam_names)],
    )
    for table, name in zip(hparam_tables[1:], hparam_names[1:]):
        hparams_stmt = hparams_stmt.join_from(
            hparam_tables[0],
            table,
            and_(
                hparam_tables[0].c.run_id == table.c.run_id,
                hparam_tables[0].c.hparam == hparam_names[0],
                table.c.hparam == name,
            ),
        )
    cte = hparams_stmt.cte()

    conds = []
    for name, values in hparam_spec.items():
        conds.append(getattr(cte.c, name).in_(values))

    raw_stmt = select(_runs).join(cte, cte.c.run_id == _runs.c.run_id)
    stmt = raw_stmt.where(and_(*conds))

    # Print result
    with _engine.connect() as conn:
        res = conn.execute(stmt).all()
    print(pd.DataFrame(res))


def flatten_tags():
    run_ids = ["run_1", "run_2"]
    stmt = select(_tags.c.tag).distinct().where(_tags.c.run_id.in_(run_ids))
    with _engine.connect() as conn:
        all_tags = [row[0] for row in conn.execute(stmt).all()]
    tags_tables = [aliased(_tags) for _ in range(len(all_tags))]
    for table, tag in zip(tags_tables, all_tags):
        pass


flatten_tags()

# Cleanup

_engine.dispose()
