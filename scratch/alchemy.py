# SQLAlchemy

from dataclasses import dataclass

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
class query:
    names: list[str]
    hparams: dict[str, str] | None = None
    collections: list[str] | None = None


specs = (
    query(
        names=["predictions_train", "predictions_val", "predictions_test"],
        hparams={"eval_logs": "log_loss", "lr": "1e-3"},
        collections=["xgboost_baseline", "xgboost_best"],
    ),
    query(names=["dataset"], collections=["german"]),
)

# SQL Query

hparams = []
for entry in specs:
    if entry.hparams is not None:
        hparams.extend(list(entry.hparams.keys()))

hs = [aliased(_hparams) for _ in range(len(hparams))]
cte = select(hs[0].c.run_id, *[h.c.value.label(n) for h, n in zip(hs, hparams)])
for h, hparam in zip(hs[1:], hparams[1:]):
    cte = cte.join_from(
        hs[0],
        h,
        and_(
            hs[0].c.run_id == h.c.run_id,
            hs[0].c.hparam == hparams[0],
            h.c.hparam == hparam,
        ),
    )
cte = cte.cte()

outer_conds = []
for entry in specs:
    inner_conds = []
    inner_conds.append(_artifacts.c.artifact_name.in_(entry.names))
    if entry.collections is not None:
        inner_conds.append(_tags.c.tag.in_(entry.collections))
    if entry.hparams is not None:
        for name, value in entry.hparams.items():
            inner_conds.append(getattr(cte.c, name) == value)
    outer_conds.append(and_(*inner_conds))

stmt = (
    select(
        _artifacts.c.run_id,
        _artifacts.c.artifact_name,
        _tags.c.tag,
        cte,
    )
    .join(_tags, _tags.c.run_id == _artifacts.c.run_id)
    .outerjoin(cte, _tags.c.run_id == cte.c.run_id)
    .where(or_(*outer_conds))
)

# Print result

with _engine.connect() as conn:
    for table in [_artifacts, _hparams, _tags]:
        conn.execute(CreateTable(table, if_not_exists=True))
    conn.commit()
    res = conn.execute(stmt).all()
print(pd.DataFrame(res))

# Cleanup

_engine.dispose()
