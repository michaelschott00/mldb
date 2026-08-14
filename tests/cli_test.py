import shutil
import tempfile
import unittest

import numpy as np
from click.testing import CliRunner

from mldb.cli import main
from mldb.store import RunStore
from tests.test_utils import Environment


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        Environment.export_variables()
        self.runner = CliRunner()
        self.store = RunStore.from_env()
        self.run_id = self.store.create_run()

    def tearDown(self) -> None:
        self.store.close()
        if Environment.MLDB_DATA_ROOT.exists():
            shutil.rmtree(Environment.MLDB_DATA_ROOT)

    def test_list(self) -> None:
        result = self.runner.invoke(main, ["list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(self.run_id, result.output)

    def test_tag_add(self) -> None:
        result = self.runner.invoke(main, ["tag", self.run_id, "+tag1"])
        self.assertEqual(result.exit_code, 0)
        matched = self.store.list_runs(tags=["tag1"])
        self.assertEqual([r.run_id for r in matched], [self.run_id])

    def test_tag_remove(self) -> None:
        self.store.add_tags(self.run_id, ["tag1"])
        result = self.runner.invoke(main, ["tag", self.run_id, "-tag1"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(self.store.list_runs(tags=["tag1"]), [])

    def test_artifacts(self) -> None:
        self.store.store(self.run_id, {"array": np.array([1, 2, 3])})
        result = self.runner.invoke(main, ["artifacts"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("array", result.output)

    def test_hparams(self) -> None:
        run_id = self.store.create_run(hparams={"lr": 0.01, "epochs": 10})
        result = self.runner.invoke(main, ["hparams", run_id])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("lr", result.output)
        self.assertIn("0.01", result.output)
        self.assertIn("epochs", result.output)
        self.assertIn("10", result.output)

    def test_merge(self) -> None:
        source_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, source_dir, ignore_errors=True)
        source_store = RunStore(root_dir=source_dir)
        source_run_id = source_store.create_run(tags=["merged"])
        source_store.store(source_run_id, {"array": np.array([1, 2, 3])})
        source_store.close()

        result = self.runner.invoke(main, ["merge", source_dir])
        self.assertEqual(result.exit_code, 0)

        matched = self.store.list_runs(tags=["merged"])
        self.assertEqual([r.run_id for r in matched], [source_run_id])
        np.testing.assert_equal(
            self.store.load(source_run_id, "array"), np.array([1, 2, 3])
        )

    def test_delete(self) -> None:
        result = self.runner.invoke(main, ["delete", self.run_id])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(f"Deleted {self.run_id}", result.output)
        run_ids = [r.run_id for r in self.store.list_runs()]
        self.assertNotIn(self.run_id, run_ids)


if __name__ == "__main__":
    unittest.main()
