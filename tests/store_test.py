import json
import os
import shutil
import unittest
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

# from PIL import Image
from db.store import RunStore


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
        _ = RunStore.from_env()
        self.assertTrue(Environment.DATA_ROOT.exists())

    def test_create_store(self):
        _ = RunStore(str(Environment.DATA_ROOT))
        self.assertTrue(Environment.DATA_ROOT.exists())

    def _create_experiment(self, tag: str) -> tuple[RunStore, str]:
        store = RunStore.from_env()
        run_id = store.create_run(tag)
        return store, run_id

    def _test_store_load(self, name: str, blob: Any, assert_func: Callable) -> None:
        store, run_id = self._create_experiment("store_" + name)
        blob_name = f"test_{name}"
        store.store(run_id, {blob_name: blob})
        recovered_blob = store.load(run_id, blob_name)
        assert_func(blob, recovered_blob)

    def test_store_table(self) -> None:
        table = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        self._test_store_load("table", table, pd.testing.assert_frame_equal)

    def test_store_numpy(self) -> None:
        array = np.array([1, 2, 3])
        self._test_store_load("array", array, np.testing.assert_equal)

    def test_store_torch(self) -> None:
        tensor = torch.Tensor([1, 2, 3])
        self._test_store_load("tensor", tensor, torch.testing.assert_close)

    def test_store_state_dict(self) -> None:
        state_dict = torch.nn.Linear(3, 2).state_dict()

        def assert_state_dict_equal(
            state_dict: dict, recovered_state_dict: dict
        ) -> None:
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
        tensor = torch.Tensor([1, 2, 3])
        array = np.array([1, 2, 3])
        store.store(run_id, {"array": array, "tensor": tensor})
        recovered_array = store.load(run_id, "array")
        recovered_tensor = store.load(run_id, "tensor")
        np.testing.assert_equal(array, recovered_array)
        torch.testing.assert_close(tensor, recovered_tensor)

    def test_store_same_object(self) -> None:
        store, run_id = self._create_experiment("store_same_object")
        array_1 = np.array([1, 2, 3])
        array_2 = np.array([1, 2, 3])
        store.store(run_id, {"array_1": array_1, "array_2": array_2})
        recovered_array_1 = store.load(run_id, "array_1")
        recovered_array_2 = store.load(run_id, "array_2")
        np.testing.assert_equal(array_1, recovered_array_1)
        np.testing.assert_equal(array_2, recovered_array_2)

    def test_store_same_reference(self) -> None:
        store, run_id = self._create_experiment("store_same_reference")
        array = np.array([1, 2, 3])
        store.store(run_id, {"array_1": array, "array_2": array})
        recovered_array_1 = store.load(run_id, "array_1")
        recovered_array_2 = store.load(run_id, "array_2")
        np.testing.assert_equal(array, recovered_array_1)
        np.testing.assert_equal(array, recovered_array_2)

    def test_store_and_query_tags(self) -> None:
        store = RunStore.from_env()
        _ = store.create_run("test_tags_1", ["tag1", "tag2", "tag3"])
        run_id_2 = store.create_run("test_tags_2", ["tag2", "tag3", "tag4"])
        _ = store.create_run("test_tags_3", ["tag3", "tag4"])
        matched_run_ids = store.query_run(["tag2", "tag3"], ["tag1"])
        self.assertEqual(len(matched_run_ids), 1)
        self.assertEqual(run_id_2, matched_run_ids[0])

    def test_load_by_tag(self) -> None:
        store = RunStore.from_env()
        df_1 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df_2 = pd.DataFrame({"x": [5, 6], "y": [7, 8]})
        df_3 = pd.DataFrame({"x": [9, 10], "y": [11, 12]})

        run_id_1 = store.create_run("run1", ["tag1", "tag2", "tag3"])
        run_id_2 = store.create_run("run2", ["tag2", "tag3", "tag4"])
        run_id_3 = store.create_run("run3", ["tag3", "tag4"])

        store.store(run_id_1, {"df": df_1})
        store.store(run_id_2, {"df": df_2})
        store.store(run_id_3, {"df": df_3})

        # include tag2+tag3, exclude tag1 → only run2 matches
        results = store.load_by_tag("df", ["tag2", "tag3"], ["tag1"])

        self.assertEqual(len(results), 1)
        self.assertTrue(run_id_2 in results)
        pd.testing.assert_frame_equal(results[run_id_2], df_2)
