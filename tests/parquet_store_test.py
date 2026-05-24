import os
import shutil
import unittest
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

from db.parquet_store import ExperimentStore


@dataclass
class Environment:
    TESTS_ROOT: Path = Path("tests")
    DATA_ROOT: Path = TESTS_ROOT / "data"

    @classmethod
    def variables(cls) -> Iterator[tuple[str, Path]]:
        return ((f.name, getattr(cls, f.name)) for f in fields(cls))

    @classmethod
    def set_variables(cls) -> None:
        for k, v in cls.variables():
            os.environ[k] = str(v)


Environment.set_variables()


class ExperimentStoreTests(unittest.TestCase):
    def _clear_dirs(self) -> None:
        for directory in [Environment.DATA_ROOT]:
            if directory.exists():
                prompt = input(f"Delete {directory}?")
                if prompt == "y":
                    shutil.rmtree(directory)

    def test_create_store_from_env(self):
        _ = ExperimentStore.from_env()
        self.assertTrue(Environment.DATA_ROOT.exists())

    def test_create_store(self):
        _ = ExperimentStore(str(Environment.DATA_ROOT))
        self.assertTrue(Environment.DATA_ROOT.exists())

    def test_store_table(self) -> None:
        store = ExperimentStore.from_env()
        run_id = store.register_run("store_table_test")
        table_name = "test_table"
        table = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        store.store({table_name: table})
        recovered_table = store.load(run_id, table_name)
        pd.testing.assert_frame_equal(table, recovered_table)

    def test_from_run(self) -> None:
        store = ExperimentStore.from_env()
        run_id = store.register_run("from_run_test")
        table_name = "test_table"
        table = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        store.store({table_name: table})
        recovered_store = ExperimentStore.from_run(run_id)
        recovered_table = recovered_store.load(run_id, table_name)
        pd.testing.assert_frame_equal(table, recovered_table)
