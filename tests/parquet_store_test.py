import os
import shutil
import unittest
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Any
import json

from PIL import Image
import torch
import numpy as np
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

    def tearDown(self) -> None:
        for directory in [Environment.DATA_ROOT]:
            if directory.exists():
                shutil.rmtree(directory)

    def test_create_store_from_env(self):
        _ = ExperimentStore.from_env()
        self.assertTrue(Environment.DATA_ROOT.exists())

    def test_create_store(self):
        _ = ExperimentStore(str(Environment.DATA_ROOT))
        self.assertTrue(Environment.DATA_ROOT.exists())

    def _create_experiment(self, tag: str) -> tuple[ExperimentStore, str]:
        store = ExperimentStore.from_env()
        run_id = store.register_run(f"{tag}_test")
        return store, run_id

    def _test_store_load(self, name: str, blob: Any, assert_func: Callable) -> None:
        store, run_id = self._create_experiment("store_" + name)
        blob_name = f"test_{name}"
        store.store({blob_name: blob})
        recovered_blob = store.load(run_id, blob_name)
        assert_func(blob, recovered_blob)

    def test_store_table(self) -> None:
        table = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        self._test_store_load("table", table, pd.testing.assert_frame_equal)

    def test_store_numpy(self) -> None:
        array = np.array([1,2,3])
        self._test_store_load("array", array, np.testing.assert_equal)

    def test_store_torch(self) -> None:
        tensor = torch.Tensor([1,2,3])
        self._test_store_load("tensor", tensor, torch.testing.assert_close)

    def test_store_state_dict(self) -> None:
        state_dict = torch.nn.Linear(3, 2).state_dict()

        def assert_state_dict_equal(state_dict: dict, recovered_state_dict: dict) -> None:
            for key in state_dict.keys():
                torch.testing.assert_close(recovered_state_dict[key], state_dict[key])

        self._test_store_load("state_dict", state_dict, assert_state_dict_equal)

    # def test_store_pil(self) -> None:
    #     image = Image.open("tests/assets/image.jpg")
    #     self._test_store_load("image", image, self.assertEqual)

    def test_store_file(self) -> None:
        with open("tests/assets/test.json", "r") as f:
            file = json.load(f)
        self._test_store_load("file", file, self.assertEqual)

    def test_store_same_data(self) -> None:
        store, run_id = self._create_experiment("store_same_data")
        tensor = torch.Tensor([1,2,3])
        array = np.array([1,2,3])
        store.store({"array": array, "tensor": tensor})
        recovered_array = store.load(run_id, "array")
        recovered_tensor = store.load(run_id, "tensor")
        np.testing.assert_equal(array, recovered_array)
        torch.testing.assert_close(tensor, recovered_tensor)
        
    def test_store_same_object(self) -> None:
        store, run_id = self._create_experiment("store_same_object")
        array_1 = np.array([1,2,3])
        array_2 = np.array([1,2,3])
        store.store({"array_1": array_1, "array_2": array_2})
        recovered_array_1 = store.load(run_id, "array_1")
        recovered_array_2 = store.load(run_id, "array_2")
        np.testing.assert_equal(array_1, recovered_array_1)
        np.testing.assert_equal(array_2, recovered_array_2)

    def test_store_same_reference(self) -> None:
        store, run_id = self._create_experiment("store_same_reference")
        array = np.array([1,2,3])
        store.store({"array_1": array, "array_2": array})
        recovered_array_1 = store.load(run_id, "array_1")
        recovered_array_2 = store.load(run_id, "array_2")
        np.testing.assert_equal(array, recovered_array_1)
        np.testing.assert_equal(array, recovered_array_2)
        
    def test_from_run(self) -> None:
        store = ExperimentStore.from_env()
        run_id = store.register_run("from_run_test")
        table_name = "test_table"
        table = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        store.store({table_name: table})
        recovered_store = ExperimentStore.from_run(run_id)
        recovered_table = recovered_store.load(run_id, table_name)
        pd.testing.assert_frame_equal(table, recovered_table)

