import json
import os
import pickle
import shutil
import unittest
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image

results_dir = "tests/results"
models_dir = "tests/models"
data_dir = "tests/data"
os.environ["DISCOVERY_RESULTS_ROOT"] = results_dir
os.environ["DISCOVERY_MODELS_ROOT"] = models_dir
os.environ["DISCOVERY_DATA_ROOT"] = data_dir

from discovery.db import (
    CUBDB,
    FMNISTDB,
    CelebADB,
    MLDataset,
    ModelDB,
    ResultsDB,
)


class DBTests(unittest.TestCase):
    def _clear_dirs(self) -> None:
        for directory in [results_dir, models_dir, data_dir]:
            assert "tests/" in directory
            if os.path.exists(directory):
                shutil.rmtree(directory)

    def setUp(self) -> None:
        assert os.getenv("DISCOVERY_RESULTS_ROOT") == results_dir
        assert os.getenv("DISCOVERY_MODELS_ROOT") == models_dir
        assert os.getenv("DISCOVERY_DATA_ROOT") == data_dir
        self._clear_dirs()
        self.results_db = ResultsDB()
        self.models_db = ModelDB()
        self.cub_db = CUBDB()  # type: ignore
        self.fmnist_db = FMNISTDB()  # type: ignore
        self.celeba_db = CelebADB()  # type: ignore
        # for directory in [results_dir, models_dir, data_dir]:
        #     print(directory)

    def tearDown(self) -> None:
        assert os.getenv("DISCOVERY_RESULTS_ROOT") == results_dir
        assert os.getenv("DISCOVERY_MODELS_ROOT") == models_dir
        assert os.getenv("DISCOVERY_DATA_ROOT") == data_dir
        self._clear_dirs()

    def test_results_register_experiment(self) -> None:
        experiment_id = self.results_db.register_run("test")
        self.assertTrue(os.path.exists(os.path.join(results_dir, experiment_id)))
        self.assertTrue(
            os.path.exists(os.path.join(results_dir, experiment_id, "blobs"))
        )
        self.assertTrue(os.path.exists(os.path.join(results_dir, "blobs")))
        # the database is only created the first time results are stored, so won't test it here

    def test_results_store_query_simple(self) -> None:
        experiment_id = self.results_db.register_run("test")
        col_a = [1, 2]
        col_b = [2.0, 3.0]
        data = pd.DataFrame({"a": col_a, "b": col_b})
        self.results_db.store_results("test", data)
        result = self.results_db.query("select * from test")
        self.assertTrue(
            os.path.exists(os.path.join(results_dir, experiment_id, "results.db"))
        )
        self.assertEqual(len(result), len(data))
        self.assertEqual(result["a"].tolist(), col_a)
        self.assertEqual(result["b"].tolist(), col_b)

    def test_results_store_query_one(self) -> None:
        self.results_db.register_run("test")
        col_a = 1.0
        col_b = 2.0
        data = pd.DataFrame({"a": [col_a], "b": [col_b]})
        self.results_db.store_results("test", data)
        result = self.results_db.query_one("select * from test")
        self.assertEqual(result["a"].tolist(), col_a)
        self.assertEqual(result["b"].tolist(), col_b)

    def test_results_store_query_one_fail(self) -> None:
        self.results_db.register_run("test")
        col_a = [1.0, 3.0]
        col_b = [2.0, 4.0]
        data = pd.DataFrame({"a": col_a, "b": col_b})
        self.results_db.store_results("test", data)
        with self.assertRaises(ValueError):
            self.results_db.query_one("select * from test")

    def test_results_store_blobs(self) -> None:
        experiment_id = self.results_db.register_run("test")

        # Create some artifacts
        artifacts: Dict[int | str, Any] = {
            "array": np.array([1, 2, 3]),
            "tensor": torch.tensor([1, 2, 3]),
            "state_dict": torch.nn.Linear(3, 2).state_dict(),
            "image": Image.open("tests/assets/image.jpg"),
            "image_file": "tests/assets/image.jpg",
            "csv": "tests/assets/test.csv",
            "json": "tests/assets/test.json",
        }

        # Store and retrieve
        self.results_db.store_artifacts(artifacts)
        retrieved_uri = dict()
        retrieved_blob = dict()
        for key in artifacts.keys():
            retrieved_uri[key] = self.results_db.query_one(
                f"""
                SELECT artifact_uri
                FROM artifacts
                WHERE artifact_name = '{key}'
                """
            ).loc["artifact_uri"]
            retrieved_blob[key] = self.results_db.load_artifact_for_checksum(
                retrieved_uri[key]
            )
        retrieved_csv = pd.read_csv(retrieved_blob["csv"])
        retrieved_json = json.load(retrieved_blob["json"])

        np.testing.assert_array_equal(retrieved_blob["array"], artifacts["array"])
        self.assertTrue(torch.equal(retrieved_blob["tensor"], artifacts["tensor"]))
        self.assertEqual(
            retrieved_blob["image_file"], Image.open("tests/assets/image.jpg")
        )
        np.testing.assert_array_equal(
            np.array(retrieved_blob["image"]),
            np.array(Image.open("tests/assets/image.jpg")),
        )
        for key in artifacts["state_dict"].keys():
            self.assertTrue(
                torch.equal(
                    retrieved_blob["state_dict"][key], artifacts["state_dict"][key]
                )
            )
        self.assertTrue(retrieved_csv.equals(pd.read_csv("tests/assets/test.csv")))
        with open("tests/assets/test.json", "r") as f:
            self.assertEqual(retrieved_json, json.load(f))

    def test_results_store_different_schema_pass(self) -> None:
        self.results_db.register_run("test")
        col_a = [1.0, 3.0]
        col_b = [2.0, 4.0]
        data_1 = pd.DataFrame({"a": col_a, "b": col_b})
        data_2 = pd.DataFrame({"b": col_b, "a": col_a})
        assert list(data_1.columns) != list(data_2.columns)
        self.results_db.store_results("test", data_1)
        self.results_db.store_results("test", data_2)
        result = self.results_db.query("select * from test")
        self.assertEqual(result["a"].tolist(), 2 * col_a)
        self.assertEqual(result["b"].tolist(), 2 * col_b)

    def test_results_store_different_schema_fail(self) -> None:
        self.results_db.register_run("test")
        col_a = [1.0, 3.0]
        col_b = [2.0, 4.0]
        col_c = ["A", "B"]
        data_1 = pd.DataFrame({"a": col_a, "b": col_b})
        data_2 = pd.DataFrame({"a": col_a, "b": col_b, "c": col_c})
        self.results_db.store_results("test", data_1)
        with self.assertRaises(Exception):
            self.results_db.store_results("test", data_2)

    def test_results_store_random_types(self) -> None:
        """Random types should get converted to strings."""
        self.results_db.register_run("test")
        col_a = [1.0, 3.0, "A"]
        col_b = [2.0, 4.0, np.array([1, 2, 3])]
        data = pd.DataFrame({"a": col_a, "b": col_b})
        self.results_db.store_results("test", data)
        result = self.results_db.query("select * from test")
        self.assertEqual(result["a"].tolist(), [str(i) for i in col_a])
        self.assertEqual(result["b"].tolist(), [str(i) for i in col_b])

    def test_models_store_model(self) -> None:
        # Values
        tag = "test"
        dataset = MLDataset.FMNIST
        complete_concepts = None
        architecture_part = "test_part"
        model = torch.nn.Linear(3, 2)
        state_dict = model.state_dict()
        np.random.seed(0)
        test_artifact = np.random.randint(low=0, high=10, size=(1, 3))
        hparam_dict = {"a": 1, "b": 2, "c": 3}
        hparams = pd.DataFrame([hparam_dict])
        artifacts: Dict[str | int, Any] = {
            "state_dict": state_dict,
            "test_artifact": test_artifact,
        }
        metrics = pd.DataFrame([{"d": 4, "e": 5, "f": 6}])
        additional_results = {"test_results": pd.DataFrame([{"g": 7, "h": 8, "i": 9}])}

        # Store
        self.models_db.register_run(tag)
        self.models_db.store_model(
            tag=tag,
            dataset=dataset,
            complete_concepts=complete_concepts,
            architecture_part=architecture_part,
            hparams=hparams,
            artifacts=artifacts,
            metrics=metrics,
            additional_results=additional_results,
        )

        for db in [self.models_db, ModelDB(self.models_db.run_id)]:
            # Retrieve
            hparam_record = db.load_hparams()
            metadata_df = db.query("select * from metadata")
            hparams_df = db.query("select * from hparams")
            metrics_df = db.query("select * from metrics")
            artifacts_df = db.query("select * from artifacts")
            additional_df = db.query("select * from test_results")

            # Check retrieved hparam recordd
            self.assertEqual(hparam_record, hparam_dict)

            # Check retrieved metadata
            self.assertEqual(metadata_df["tag"].tolist(), [tag])
            self.assertEqual(metadata_df["dataset"].tolist(), [dataset.value])
            self.assertTrue(
                metadata_df["complete_concepts"].isna().all(),
                metadata_df["complete_concepts"],
            )
            self.assertEqual(
                metadata_df["architecture_part"].tolist(), [architecture_part]
            )

            # Check retrieved hparams
            self.assertEqual(hparams_df["a"].tolist(), [1])
            self.assertEqual(hparams_df["b"].tolist(), [2])
            self.assertEqual(hparams_df["c"].tolist(), [3])

            # Check retrieved metrics
            self.assertEqual(metrics_df["d"].tolist(), [4])
            self.assertEqual(metrics_df["e"].tolist(), [5])
            self.assertEqual(metrics_df["f"].tolist(), [6])

            # Check retrieved artifacts
            self.assertEqual(
                artifacts_df["artifact_name"].tolist(), ["state_dict", "test_artifact"]
            )
            restored_state_dict = db.load_artifact_for_checksum(
                artifacts_df["artifact_uri"].iloc[0]
            )
            for key in state_dict.keys():
                self.assertTrue(torch.equal(restored_state_dict[key], state_dict[key]))
            restored_test_artifact = db.load_artifact_for_checksum(
                artifacts_df["artifact_uri"].iloc[1]
            )
            self.assertTrue(
                np.array_equal(restored_test_artifact, test_artifact),
                f"retrieved: {restored_test_artifact}, expected: {test_artifact}",
            )

            # Check retrieved additional results
            self.assertEqual(additional_df["g"].tolist(), [7])
            self.assertEqual(additional_df["h"].tolist(), [8])
            self.assertEqual(additional_df["i"].tolist(), [9])

    def test_data_setup(self) -> None:
        for name in ["cub", "fmnist", "celeba"]:
            self.assertTrue(os.path.exists(os.path.join(data_dir, name)))
            self.assertTrue(os.path.exists(os.path.join(data_dir, name, "blobs")))

    def test_store_images(self) -> None:
        # Create random labels for each dataset
        images: Dict[int | str, Any] = {
            1: "tests/assets/image.jpg",
            2: "tests/assets/image.jpg",
        }
        labels = {
            "cub": pd.DataFrame(
                {
                    "img_id": list(images.keys()),
                    "class_id": [np.random.randint(0, 100) for _ in range(2)],
                },
            ),
            "fmnist": pd.DataFrame(
                {
                    "img_id": list(images.keys()),
                    "class_id": [np.random.randint(0, 100) for _ in range(2)],
                }
            ),
            "celeba": pd.DataFrame(
                {
                    "img_id": list(images.keys()),
                    "class_id": [np.random.randint(0, 100) for _ in range(2)],
                }
            ),
        }

        # Test datasets
        for db, db_name in zip(
            [self.cub_db, self.fmnist_db, self.celeba_db], labels.keys()
        ):
            db.store_images(images)
            df = labels[db_name]
            db.store_image_class_labels(df)
            result_image_table = db.query("select * from images")
            result_labels = db.query("select class_id from image_class_labels")
            self.assertTrue(set(result_image_table.columns) == {"image", "img_id"})
            self.assertEqual(len(result_image_table), len(images))
            for image_path, image_uri in zip(
                images.values(), result_image_table["image"]
            ):
                original_image = Image.open(image_path)
                retrieved_image = db.load_artifact_for_checksum(image_uri)
                self.assertTrue(
                    np.array_equal(np.array(original_image), np.array(retrieved_image))
                )
            self.assertEqual(
                result_labels["class_id"].tolist(), df["class_id"].tolist()
            )


if __name__ == "__main__":
    unittest.main()
