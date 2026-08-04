from __future__ import annotations

from typing import Any

from jsonargparse import Namespace
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from mldb.store import RunStore


def flatten_hparams(config: Any, parent_key: str = "") -> dict[str, Any]:
    """Flattens a (possibly nested) jsonargparse Namespace/dict into dotted-key hparams."""
    if isinstance(config, Namespace):
        config = vars(config)
    if isinstance(config, dict):
        hparams = {}
        for key, value in config.items():
            dotted_key = f"{parent_key}.{key}" if parent_key else key
            hparams.update(flatten_hparams(value, dotted_key))
        return hparams
    if isinstance(config, type):
        return {parent_key: config.__name__}
    if type(config).__repr__ is object.__repr__:
        return {parent_key: config.__class__.__name__}
    return {parent_key: config}


def configure_run(config: Any, results_dir: str | None = None) -> tuple[RunStore, str]:
    """Create an mldb run from a LightningCLI (sub)config, recording it as the run's hparams
    and pointing `config.trainer.logger` at a TensorBoardLogger rooted in the run's mldb directory.

    Returns the store and run_id so callers can attach them (e.g. to the trainer) or use them
    later for storing artifacts.
    """
    store = (
        RunStore(root_dir=results_dir)
        if results_dir is not None
        else RunStore.from_env()
    )
    hparams = flatten_hparams(config)
    run_id = store.create_run(hparams=hparams)
    uri = store.open_directory(run_id)
    config.trainer.logger = {
        "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
        "init_args": {"save_dir": uri, "name": "", "version": ""},
    }
    return store, run_id


class MLDBLightningCLI(LightningCLI):
    """LightningCLI that wires up an mldb run via `configure_run` before Lightning classes are instantiated.

    Adds a `--results_dir` argument for pointing at a non-default mldb store, and exposes the
    resulting `store`/`run_id` on the trainer as `trainer.store`/`trainer.run_id`.
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--results_dir",
            type=str,
            default=None,
            help="Root directory for mldb run storage. Falls back to RunStore.from_env() if unset.",
        )

    def before_instantiate_classes(self) -> None:
        config = self.config[self.subcommand] if self.subcommand else self.config
        self.store, self.run_id = configure_run(config, config.results_dir)

    def after_instantiate_classes(self) -> None:
        self.trainer.store = self.store
        self.trainer.run_id = self.run_id
