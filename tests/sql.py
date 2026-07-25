import duckdb
import pandas as pd

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

# SQL

# TODO: translate to sql alchemy
with duckdb.connect(":memory:") as con:
    con.execute("create table tags as (select * from tag_df)")
    con.execute("create table artifacts as (select * from artifact_df)")
    con.execute("create table hparams as (select * from hparam_df)")
    res = con.sql("""
        with hparams_flat as (
            select h1.run_id, h1.value as eval_logs, h2.value as lr
            from hparams h1
            join hparams h2 on h1.run_id=h2.run_id and h1.hparam='eval_logs' and h2.hparam='lr'
            )
        select a.run_id, a.artifact_name, t.tag, h.eval_logs, h.lr
        from tags t
        join artifacts a on t.run_id=a.run_id
        left outer join hparams_flat h on t.run_id=h.run_id
        where (
            tag in ['xgboost_baseline', 'xgboost_best'] and
            eval_logs='log_loss' and lr='1e-3'
            and artifact_name in ['predictions_train', 'predictions_val', 'predictions_test']
        ) or (
            artifact_name='dataset' and tag='german'
            )
    """)
    print(res)
