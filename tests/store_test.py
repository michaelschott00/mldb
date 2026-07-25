import json
import multiprocessing
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
from mldb.store import RunStore


@dataclass
class Environment:
    TESTS_ROOT: Path = Path("tests")
    DATA_ROOT: Path = TESTS_ROOT / "test_data"

    @classmethod
    def variables(cls) -> Iterator[tuple[str, Path]]:
        return ((f.name, getattr(cls, f.name)) for f in fields(cls))

    @classmethod
    def set_variables(cls) -> None:
        for k, v in cls.variables():
            os.environ[k] = str(v)


Environment.set_variables()


def _concurrent_worker(
    root_dir: str, data: list, result_queue: multiprocessing.Queue
) -> None:
    store = RunStore(root_dir)
    run_id = store.create_run()
    store.store(run_id, {"array": np.array(data)})
    result_queue.put((run_id, data))


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

    def _create_experiment(self) -> tuple[RunStore, str]:
        store = RunStore.from_env()
        run_id = store.create_run()
        return store, run_id

    def _test_store_load(self, name: str, blob: Any, assert_func: Callable) -> None:
        store, run_id = self._create_experiment()
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
        store, run_id = self._create_experiment()
        tensor = torch.Tensor([1, 2, 3])
        array = np.array([1, 2, 3])
        store.store(run_id, {"array": array, "tensor": tensor})
        recovered_array = store.load(run_id, "array")
        recovered_tensor = store.load(run_id, "tensor")
        np.testing.assert_equal(array, recovered_array)
        torch.testing.assert_close(tensor, recovered_tensor)

    def test_store_same_object(self) -> None:
        store, run_id = self._create_experiment()
        array_1 = np.array([1, 2, 3])
        array_2 = np.array([1, 2, 3])
        store.store(run_id, {"array_1": array_1, "array_2": array_2})
        recovered_array_1 = store.load(run_id, "array_1")
        recovered_array_2 = store.load(run_id, "array_2")
        np.testing.assert_equal(array_1, recovered_array_1)
        np.testing.assert_equal(array_2, recovered_array_2)

    def test_store_same_reference(self) -> None:
        store, run_id = self._create_experiment()
        array = np.array([1, 2, 3])
        store.store(run_id, {"array_1": array, "array_2": array})
        recovered_array_1 = store.load(run_id, "array_1")
        recovered_array_2 = store.load(run_id, "array_2")
        np.testing.assert_equal(array, recovered_array_1)
        np.testing.assert_equal(array, recovered_array_2)

    def test_add_tags_after_run_creation(self) -> None:
        store = RunStore.from_env()
        run_id = store.create_run()
        store.add_tags(run_id, ["tag1", "tag2"])
        matched_run_ids = [r.run_id for r in store.list_runs(["tag1", "tag2"], [])]
        self.assertEqual(len(matched_run_ids), 1)
        self.assertEqual(matched_run_ids[0], run_id)

    def test_store_and_query_tags(self) -> None:
        store = RunStore.from_env()
        _ = store.create_run(["tag1", "tag2", "tag3"])
        run_id_2 = store.create_run(["tag2", "tag3", "tag4"])
        _ = store.create_run(["tag3", "tag4"])
        matched_run_ids = [
            r.run_id for r in store.list_runs(["tag2", "tag3"], ["tag1"])
        ]
        self.assertEqual(len(matched_run_ids), 1)
        self.assertEqual(run_id_2, matched_run_ids[0])

    def test_load_by_tags(self) -> None:
        store = RunStore.from_env()
        df_1 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df_2 = pd.DataFrame({"x": [5, 6], "y": [7, 8]})
        df_3 = pd.DataFrame({"x": [9, 10], "y": [11, 12]})

        run_id_1 = store.create_run(["tag1", "tag2", "tag3"])
        run_id_2 = store.create_run(["tag2", "tag3", "tag4"])
        run_id_3 = store.create_run(["tag3", "tag4"])

        store.store(run_id_1, {"df": df_1})
        store.store(run_id_2, {"df": df_2})
        store.store(run_id_3, {"df": df_3})

        # include tag2+tag3, exclude tag1 → only run2 matches
        results = store.load_artifacts_by_tags("df", ["tag2", "tag3"], ["tag1"])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].run_id, run_id_2)
        pd.testing.assert_frame_equal(results[0].artifact, df_2)

    def test_delete_run(self) -> None:
        store = RunStore.from_env()
        run_id = store.create_run(["delete_tag"])
        array = np.array([1, 2, 3])
        store.store(run_id, {"array": array})

        store.delete_run(run_id)

        self.assertEqual(len(store.list_runs(["delete_tag"], [])), 0)
        with self.assertRaises(ValueError):
            store.load(run_id, "array")

    def test_delete_run_preserves_shared_artifact(self) -> None:
        store = RunStore.from_env()
        array = np.array([1, 2, 3])
        run_id_1 = store.create_run()
        run_id_2 = store.create_run()
        store.store(run_id_1, {"array": array})
        store.store(run_id_2, {"array": array})

        store.delete_run(run_id_1)

        np.testing.assert_equal(store.load(run_id_2, "array"), array)

    def test_concurrent_run_creation(self) -> None:
        root_dir = str(Environment.DATA_ROOT)
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        processes = [
            multiprocessing.Process(
                target=_concurrent_worker, args=(root_dir, [1, 2, 3], result_queue)
            ),
            multiprocessing.Process(
                target=_concurrent_worker, args=(root_dir, [4, 5, 6], result_queue)
            ),
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
            self.assertEqual(p.exitcode, 0)

        store = RunStore(root_dir)
        for _ in processes:
            run_id, data = result_queue.get()
            np.testing.assert_equal(store.load(run_id, "array"), np.array(data))

    def _prepare_workflow_test_data(
        self,
    ) -> tuple[list[str], str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        img_ids = [1, 2, 3, 4]
        df_an_1 = pd.DataFrame(
            {
                "img_id": img_ids * 3,
                "attribute_id": [1] * 4 + [2] * 4 + [3] * 4,
                "is_present": [
                    True,
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            }
        )
        df_an_2 = pd.DataFrame(
            {
                "img_id": img_ids * 3,
                "attribute_id": [1] * 4 + [2] * 4 + [3] * 4,
                "is_present": [
                    False,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            }
        )
        df_m = pd.DataFrame(
            {"img_id": img_ids, "measurement": [i / 10 for i in range(len(img_ids))]}
        )

        # Initialize multiple datasets
        ds_run_ids = list()
        for i, df_an in enumerate([df_an_1, df_an_2]):
            store = RunStore.from_env()
            tags = [f"dataset_{i + 1}"]
            if i == 0:
                tags.append("workflow_demo")
            ds_run_id = store.create_run(tags)
            ds_run_ids.append(ds_run_id)
            store.store(ds_run_id, {"attribute_names": df_an})

        # Run experiment on dataset 1
        store = RunStore.from_env()
        exp_run_id = store.create_run(["my_tuning_run", "workflow_demo"])
        store.store(exp_run_id, {"measurements": df_m})

        df = df_an_1.merge(df_m, on="img_id")

        return ds_run_ids, exp_run_id, df_an_1, df_an_2, df_m, df

    def test_dataset_pandas_workflow(self) -> None:
        ds_run_ids, exp_run_id, _, _, _, df = self._prepare_workflow_test_data()

        # Analyze results together with dataset metadata
        store = RunStore.from_env()
        df_an_rec = store.load_artifact_by_tags(
            "attribute_names", ["dataset_1"], []
        ).artifact
        df_m_rec = store.load_artifact_by_tags(
            "measurements", ["my_tuning_run"], []
        ).artifact
        df_rec = df_an_rec.merge(df_m_rec, on="img_id")
        pd.testing.assert_frame_equal(df_rec, df)

    def test_dataset_duckdb_workflow(self) -> None:
        _, _, _, _, _, df = self._prepare_workflow_test_data()

        # Analyze results together with dataset metadata
        store = RunStore.from_env()
        with store.load_duckdb(include_tags=["workflow_demo"]) as con:
            df_rec = con.sql(
                "select * exclude m.img_id from attribute_names n join measurements m on n.img_id=m.img_id"
            ).df()
        pd.testing.assert_frame_equal(df_rec, df)


if __name__ == "__main__":
    unittest.main()
