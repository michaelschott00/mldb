import json
import multiprocessing
import os
import shutil
import tempfile
import unittest
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
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
    MLDB_DATA_ROOT: Path = TESTS_ROOT / "test_data"

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
    # Pytest

    def tearDown(self) -> None:
        for directory in [Environment.MLDB_DATA_ROOT]:
            if directory.exists():
                shutil.rmtree(directory)

    # Creation tests

    def test_create_store_from_env(self):
        _ = RunStore.from_env()
        self.assertTrue(Environment.MLDB_DATA_ROOT.exists())

    def test_create_store(self):
        _ = RunStore(str(Environment.MLDB_DATA_ROOT))
        self.assertTrue(Environment.MLDB_DATA_ROOT.exists())

    def test_concurrent_run_creation(self) -> None:
        root_dir = str(Environment.MLDB_DATA_ROOT)
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

    # Store-Load tests

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

    def test_open_directory(self) -> None:
        store, run_id = self._create_experiment()
        run_dir = store.open_directory(run_id)
        run_dir_path = Path(run_dir)
        test_str = "test"
        with open(run_dir_path / "test.txt", "w") as f:
            f.write(test_str)
        recovered_run_dir = store.open_directory(run_id)
        recovered_run_dir_path = Path(recovered_run_dir)
        with open(recovered_run_dir_path / "test.txt", "r") as f:
            recovered_str = f.read()
        self.assertEqual(test_str, recovered_str)

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

    # Modifying metadata tests

    def test_add_tags_after_run_creation(self) -> None:
        store = RunStore.from_env()
        run_id = store.create_run()
        store.add_tags(run_id, ["tag1", "tag2"])
        matched_run_ids = [r.run_id for r in store.list_runs(tags=["tag1", "tag2"])]
        self.assertEqual(len(matched_run_ids), 1)
        self.assertEqual(matched_run_ids[0], run_id)

    def test_add_hparams_after_run_creation(self) -> None:
        store = RunStore.from_env()
        run_id = store.create_run()
        store.add_hparams(run_id, {"hparam_1": 3, "hparam_2": "v2"})
        matched_run_ids = [
            r.run_id
            for r in store.list_runs(hparams={"hparam_1": [3], "hparam_2": ["v2"]})
        ]
        self.assertEqual(len(matched_run_ids), 1)
        self.assertEqual(matched_run_ids[0], run_id)

    # Query tests

    @dataclass
    class QueryTestData:
        df_1 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df_2 = pd.DataFrame({"x": [5, 6], "y": [7, 8]})
        df_3 = pd.DataFrame({"x": [9, 10], "y": [11, 12]})
        run_id_1: str = field(init=False)
        run_id_2: str = field(init=False)
        run_id_3: str = field(init=False)
        store: RunStore = field(init=False)

        def __post_init__(self) -> None:
            self.store = RunStore.from_env()
            self.run_id_1 = self.store.create_run(
                hparams={"hparam_1": 1, "hparam_2": "v1"},
                tags=["tag1", "tag2", "tag3"],
            )
            self.run_id_2 = self.store.create_run(
                hparams={"hparam_1": 2e-2, "hparam_2": "v2"},
                tags=["tag3", "tag4", "tag5"],
            )
            self.run_id_3 = self.store.create_run(
                hparams={"hparam_1": -0.3, "hparam_2": "v3"},
                tags=["tag5", "tag6", "tag7"],
            )
            self.store.store(self.run_id_1, {"df": self.df_1})
            self.store.store(self.run_id_2, {"df": self.df_2})
            self.store.store(self.run_id_3, {"df": self.df_3})

    def test_store_and_query_tags(self) -> None:
        test_data = ExperimentStoreTests.QueryTestData()
        matched_run_ids = [
            r.run_id for r in test_data.store.list_runs(tags=["tag2", "tag3"])
        ]
        self.assertEqual(len(matched_run_ids), 2)
        self.assertEqual(
            sorted([test_data.run_id_1, test_data.run_id_2]), sorted(matched_run_ids)
        )

    def test_store_and_query_hparams(self) -> None:
        test_data = ExperimentStoreTests.QueryTestData()
        matched_run_ids = [
            r.run_id
            for r in test_data.store.list_runs(hparams={"hparam_1": [-0.3, 0.02]})
        ]
        self.assertEqual(len(matched_run_ids), 2)
        self.assertEqual(
            sorted([test_data.run_id_2, test_data.run_id_3]), sorted(matched_run_ids)
        )

    def test_query_single_hparam_ignores_other_hparam_values(self) -> None:
        store = RunStore.from_env()
        run_id = store.create_run(hparams={"hparam_1": "v1", "hparam_2": "v2"})
        matched_run_ids = [
            r.run_id for r in store.list_runs(hparams={"hparam_1": ["v2"]})
        ]
        self.assertEqual(matched_run_ids, [])

    def test_store_and_query_empty(self) -> None:
        test_data = ExperimentStoreTests.QueryTestData()
        matched_run_ids_empty = sorted(
            [r.run_id for r in test_data.store.list_runs(tags=list(), hparams=dict())]
        )
        matched_run_ids_none = sorted(
            [r.run_id for r in test_data.store.list_runs(tags=None, hparams=None)]
        )
        self.assertEqual(len(matched_run_ids_empty), 3)
        self.assertEqual(len(matched_run_ids_none), 3)
        self.assertEqual(
            sorted([test_data.run_id_1, test_data.run_id_2, test_data.run_id_3]),
            matched_run_ids_empty,
        )
        self.assertEqual(matched_run_ids_empty, matched_run_ids_none)

    def test_list_artifacts_by_tags(self) -> None:
        test_data = ExperimentStoreTests.QueryTestData()
        results = test_data.store.list_artifacts_by_query(
            names=["df"], tags=["tag2", "tag6"]
        )
        retrieved_run_ids = {r.run_id for r in results}
        self.assertEqual(len(results), 2)
        self.assertTrue(test_data.run_id_1 in retrieved_run_ids)
        self.assertTrue(test_data.run_id_3 in retrieved_run_ids)

    def test_list_artifacts_by_hparams(self) -> None:
        test_data = ExperimentStoreTests.QueryTestData()
        results = test_data.store.list_artifacts_by_query(
            names=["df"], hparams={"hparam_1": [1, 2e-2]}
        )
        retrieved_run_ids = [r.run_id for r in results]
        retrieved_artifacts = [r.artifact_name for r in results]
        self.assertEqual(len(results), 2)
        self.assertEqual(len(retrieved_artifacts), 2)
        self.assertTrue(test_data.run_id_1 in retrieved_run_ids)
        self.assertTrue(test_data.run_id_2 in retrieved_run_ids)
        self.assertTrue(all(v == "df" for v in retrieved_artifacts))

    def test_list_artifacts_by_hparams_and_tags(self) -> None:
        test_data = ExperimentStoreTests.QueryTestData()
        results = test_data.store.list_artifacts_by_query(
            names=["df"], tags=["tag1", "tag4"], hparams={"hparam_1": [1, 2]}
        )
        retrieved_run_ids = [r.run_id for r in results]
        retrieved_artifacts = [r.artifact_name for r in results]
        self.assertEqual(len(results), 1)
        self.assertEqual(len(retrieved_artifacts), 1)
        self.assertTrue(test_data.run_id_1 in retrieved_run_ids)
        self.assertTrue(all(v == "df" for v in retrieved_artifacts))

    def test_load_by_tags(self) -> None:
        test_data = ExperimentStoreTests.QueryTestData()
        results = test_data.store.load_artifacts_by_query(
            names=["df"], tags=["tag1", "tag7"]
        )
        retrieved_run_ids = {r.run_id for r in results}
        self.assertEqual(len(results), 2)
        self.assertTrue(test_data.run_id_1 in retrieved_run_ids)
        self.assertTrue(test_data.run_id_3 in retrieved_run_ids)
        for r in results:
            if r.run_id == test_data.run_id_1:
                pd.testing.assert_frame_equal(r.artifact, test_data.df_1)
            if r.run_id == test_data.run_id_3:
                pd.testing.assert_frame_equal(r.artifact, test_data.df_3)

    def test_load_by_hparams(self) -> None:
        test_data = ExperimentStoreTests.QueryTestData()
        results = test_data.store.load_artifacts_by_query(
            names=["df"], tags=["tag4", "tag6"]
        )
        retrieved_run_ids = {r.run_id for r in results}
        self.assertEqual(len(results), 2)
        self.assertTrue(test_data.run_id_2 in retrieved_run_ids)
        self.assertTrue(test_data.run_id_3 in retrieved_run_ids)
        for r in results:
            if r.run_id == test_data.run_id_2:
                pd.testing.assert_frame_equal(r.artifact, test_data.df_2)
            if r.run_id == test_data.run_id_3:
                pd.testing.assert_frame_equal(r.artifact, test_data.df_3)

    # Deletion tests

    def test_delete_run(self) -> None:
        store = RunStore.from_env()
        run_id = store.create_run(tags=["delete_tag"])
        array = np.array([1, 2, 3])
        store.store(run_id, {"array": array})
        store.delete_run(run_id)
        self.assertEqual(len(store.list_runs(tags=["delete_tag"])), 0)
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

    # Merge tests

    def _create_source_store(self, **kwargs) -> tuple[RunStore, str]:
        source_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, source_dir, ignore_errors=True)
        return RunStore(root_dir=source_dir, **kwargs), source_dir

    def test_merge_runs_tags_hparams_artifacts(self) -> None:
        source_store, source_dir = self._create_source_store()
        array = np.array([1, 2, 3])
        run_id = source_store.create_run(hparams={"hparam_1": 1}, tags=["tag1", "tag2"])
        source_store.store(run_id, {"array": array})
        source_store.close()

        dest_store = RunStore.from_env()
        dest_store.merge(source_dir)

        runs = dest_store.list_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].run_id, run_id)
        self.assertEqual(runs[0].tags, ["tag1", "tag2"])
        self.assertEqual(runs[0].hparams, ["hparam_1=1"])
        np.testing.assert_equal(dest_store.load(run_id, "array"), array)

    def test_merge_copies_directory_artifacts(self) -> None:
        source_store, source_dir = self._create_source_store()
        run_id = source_store.create_run()
        run_dir = Path(source_store.open_directory(run_id))
        with open(run_dir / "test.txt", "w") as f:
            f.write("hello")
        source_store.close()

        dest_store = RunStore.from_env()
        dest_store.merge(source_dir)

        merged_dir = Path(dest_store.list_directory_by_run(run_id))
        with open(merged_dir / "test.txt", "r") as f:
            self.assertEqual(f.read(), "hello")

    def test_merge_is_idempotent(self) -> None:
        source_store, source_dir = self._create_source_store()
        array = np.array([1, 2, 3])
        run_id = source_store.create_run(tags=["tag1"])
        source_store.store(run_id, {"array": array})
        source_store.close()

        dest_store = RunStore.from_env()
        dest_store.merge(source_dir)
        dest_store.merge(source_dir)

        runs = dest_store.list_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].tags, ["tag1"])
        np.testing.assert_equal(dest_store.load(run_id, "array"), array)

    def test_merge_preserves_existing_dest_runs(self) -> None:
        source_store, source_dir = self._create_source_store()
        source_run_id = source_store.create_run()
        source_store.close()

        dest_store = RunStore.from_env()
        dest_run_id = dest_store.create_run()
        dest_store.merge(source_dir)

        run_ids = {r.run_id for r in dest_store.list_runs()}
        self.assertEqual(run_ids, {source_run_id, dest_run_id})

    def test_merge_does_not_modify_source_directory(self) -> None:
        source_store, source_dir = self._create_source_store()
        array = np.array([1, 2, 3])
        run_id = source_store.create_run()
        source_store.store(run_id, {"array": array})
        source_store.close()

        files_before = sorted(
            str(p.relative_to(source_dir)) for p in Path(source_dir).rglob("*")
        )

        dest_store = RunStore.from_env()
        dest_store.merge(source_dir)

        files_after = sorted(
            str(p.relative_to(source_dir)) for p in Path(source_dir).rglob("*")
        )
        self.assertEqual(files_before, files_after)

    def test_merge_raises_on_hash_settings_mismatch(self) -> None:
        source_store, source_dir = self._create_source_store(
            hash_depth=2, hash_granularity=2
        )
        source_store.create_run()
        source_store.close()

        dest_store = RunStore.from_env()
        with self.assertRaises(ValueError):
            dest_store.merge(source_dir)

    # Full workflow tests

    @dataclass
    class WorkflowTestData:
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
        store: RunStore = field(init=False)
        df: pd.DataFrame = field(init=False)

        def __post_init__(self) -> None:
            self.store = RunStore.from_env()

            # Initialize multiple datasets
            ds_run_ids = list()
            for i, df_an in enumerate([self.df_an_1, self.df_an_2]):
                tags = [f"dataset_{i + 1}"]
                if i == 0:
                    tags.append("workflow_demo")
                ds_run_id = self.store.create_run(tags=tags)
                ds_run_ids.append(ds_run_id)
                self.store.store(ds_run_id, {"attribute_names": df_an})

            # Run experiment on dataset 1
            exp_run_id = self.store.create_run(tags=["my_tuning_run", "workflow_demo"])
            self.store.store(exp_run_id, {"measurements": self.df_m})

            # Create final dataframe for analysis
            self.df = self.df_an_1.merge(self.df_m, on="img_id")

    def test_dataset_pandas_workflow(self) -> None:
        test_data = ExperimentStoreTests.WorkflowTestData()

        # Analyze results together with dataset metadata
        df_an_rec = test_data.store.load_artifact_by_query(
            name="attribute_names", tags=["dataset_1"]
        )
        df_m_rec = test_data.store.load_artifact_by_query(
            name="measurements", tags=["my_tuning_run"]
        )
        df_rec = df_an_rec.merge(df_m_rec, on="img_id")
        pd.testing.assert_frame_equal(df_rec, test_data.df)

    def test_dataset_duckdb_workflow(self) -> None:
        test_data = ExperimentStoreTests.WorkflowTestData()

        # Build a database containing results together with the input dataset
        db = test_data.store.get_db()
        db.attach(names=["attribute_names"], tags=["dataset_1"])
        db.attach(names=["measurements"], tags=["my_tuning_run"])
        with db.connect() as con:
            df_rec = con.sql(
                "select n.img_id, n.attribute_id, n.is_present, m.measurement "
                "from attribute_names n join measurements m on n.img_id=m.img_id"
            ).df()
            attribute_names_cols = set(con.sql("select * from attribute_names").columns)
            measurements_cols = set(con.sql("select * from measurements").columns)
        pd.testing.assert_frame_equal(df_rec, test_data.df)

        # Tags of each attached run should be joined in as columns on every table
        assert {"run_id", "dataset_1", "workflow_demo"}.issubset(attribute_names_cols)
        assert {"run_id", "my_tuning_run", "workflow_demo"}.issubset(measurements_cols)


if __name__ == "__main__":
    unittest.main()
