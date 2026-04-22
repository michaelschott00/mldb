"""My data management solution."""

import argparse
import glob
import hashlib
import io
import os
import pickle
import random
import re
import shutil
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum, auto
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Self,
    Sequence,
    Set,
    Tuple,
    Type,
    TypedDict,
)
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pandera.pandas as pa
import torch
from datasets import Dataset, Image, load_from_disk
from PIL import Image as PILImage

from discovery.registry import Registry, TypeHandlerRegistry
from discovery.utils import get_num_cpu_workers

DEFAULT_HASH_DEPTH = 3
DEFAULT_HASH_GRANULARITY = 1
IDField = partial(pa.Field, ge=1, coerce=True)


class StorableBlob(ABC):
    ext = None

    @classmethod
    @abstractmethod
    def to_str(cls, blob: Any) -> str:
        raise NotImplementedError

    @classmethod
    def checksum(cls, blob: Any) -> str:
        return hashlib.sha256(cls.to_bytes(blob)).hexdigest()

    @classmethod
    def create_hash(cls, blob: Any) -> str:
        content_hash = cls.checksum(blob)
        readable_name = (
            re.sub(r"[^_A-Za-z0-9]+", "_", cls.to_str(blob)).lower().rstrip("_")
        )
        return f"{content_hash}_{readable_name}"

    @classmethod
    @abstractmethod
    def to_bytes(cls, blob: Any) -> bytes:
        raise NotImplementedError

    @classmethod
    def prepare(cls, blob: Any) -> Any:
        return blob

    @classmethod
    def save_to_disk(cls, blob: Any, path: str) -> None:
        path = cls._add_file_extension(path)
        cls._save_operation(blob, path)

    @classmethod
    @abstractmethod
    def load_from_disk(cls, path: str, *args, **kwargs) -> Any:
        # Shouldn't handle file extensions, because that is assumed to be present in the path
        raise NotImplementedError

    @classmethod
    def _add_file_extension(cls, path: str) -> str:
        name, ext = os.path.splitext(path)
        if cls.ext is not None:
            if ext != "" and ext != cls.ext:
                warnings.warn(f"Overriding file extension {ext} with {cls.ext}")
            path = name + cls.ext
        else:
            if ext == "":
                raise ValueError(
                    f"{path} in {cls.__name__} doesn't specify a file extension"
                )
        return path

    @classmethod
    @abstractmethod
    def _save_operation(cls, blob: Any, path: str) -> None:
        raise NotImplementedError


store_handlers = TypeHandlerRegistry[StorableBlob]("StoreRegistry")
load_handlers = Registry[StorableBlob, str]("LoadRegistry")


@store_handlers.register(np.ndarray)
@load_handlers.register(".npy")
class NumpyBlob(StorableBlob):
    ext = ".npy"

    @classmethod
    def to_bytes(cls, blob: np.ndarray) -> bytes:
        return blob.tobytes()

    @classmethod
    def load_from_disk(cls, path: str, *args, **kwargs) -> np.ndarray:
        if "allow_pickle" in kwargs:
            warnings.warn("Argument allow_pickle is set to True automatically.")
            kwargs["allow_pickle"] = True
        return np.load(path, *args, **kwargs)

    @classmethod
    def to_str(cls, blob: np.ndarray) -> str:
        return f"Numpy({list(blob.shape)}, {str(blob.dtype)})"

    @classmethod
    def _save_operation(cls, blob: np.ndarray, path: str) -> None:
        np.save(path, blob)


# Introduced to get around non-determinism in PIL.Image.save for jpg.
# Now also used for other files, such as csv or JSON.
@store_handlers.register(str)
@load_handlers.register(".txt")
@load_handlers.register(".csv")
@load_handlers.register(".json")
class FileBlob(StorableBlob):
    """Copies arbitrary files to the blob store."""

    @classmethod
    def to_bytes(cls, blob: str) -> bytes:
        byte_str = cls.load_from_disk(blob)
        return byte_str.read()

    @classmethod
    def load_from_disk(cls, path: str, *args, **kwargs) -> io.BytesIO:
        assert args == () and kwargs == {}, (
            f"load_from_disk does not accept args or kwargs but got {args} and {kwargs}."
        )
        with open(path, "rb") as f:
            return io.BytesIO(f.read())

    @classmethod
    def to_str(cls, blob: str) -> str:
        basename = os.path.basename(blob)
        return f"File({basename})"

    @classmethod
    def _add_file_extension(cls, path: str) -> str:
        filename_parts = path.split("_")
        ext = filename_parts[-1]
        return f"{path}.{ext}"

    @classmethod
    def _save_operation(cls, blob: str, path: str) -> None:
        shutil.copyfile(blob, path)


@store_handlers.register(PILImage.Image)
@load_handlers.register(".jpg")
@load_handlers.register(".png")
class ImageBlob(StorableBlob):
    ext = ".png"

    @classmethod
    def to_bytes(cls, blob: PILImage.Image) -> bytes:
        return blob.tobytes()

    @classmethod
    def load_from_disk(cls, path: str, *args, **kwargs) -> PILImage.Image:
        image = PILImage.open(path, *args, **kwargs)
        if kwargs.get("rgb", False):
            return image.convert("RGB")
        return image

    @classmethod
    def to_str(cls, blob: PILImage.Image) -> str:
        return f"Image({blob.size}, {blob.mode})"

    @classmethod
    def _save_operation(cls, blob: PILImage.Image, path: str) -> None:
        """Only supports png images for now, since the checksum is deterministic."""
        blob.save(path)


@store_handlers.register(torch.Tensor)
@load_handlers.register(".pth")
class TensorBlob(StorableBlob):
    ext = ".pth"

    @classmethod
    def prepare(cls, blob: torch.Tensor) -> torch.Tensor:
        return blob.cpu()

    @classmethod
    def to_bytes(cls, blob: torch.Tensor) -> bytes:
        return blob.numpy().tobytes()

    @classmethod
    def load_from_disk(cls, path: str, *args, **kwargs) -> torch.Tensor:
        return torch.load(path, *args, **kwargs)

    @classmethod
    def to_str(cls, blob: torch.Tensor) -> str:
        return f"Tensor({list(blob.shape)}, {str(blob.dtype)})"

    @classmethod
    def _save_operation(cls, blob: torch.Tensor, path: str) -> None:
        torch.save(blob, path)


# Needed despite the TensorBlob because hashes are computed differently.
@store_handlers.register(dict)
@load_handlers.register(".state_dict")
class TorchStateDictBlob(StorableBlob):
    ext = ".state_dict"

    @classmethod
    def prepare(cls, blob: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v.cpu() if hasattr(v, "cpu") else v for k, v in blob.items()}

    @classmethod
    def load_from_disk(cls, path: str, *args, **kwargs) -> Dict[str, Any]:
        return torch.load(path, *args, **kwargs)

    @classmethod
    def to_bytes(cls, blob: Dict[str, Any]) -> bytes:
        buffer = io.BytesIO()
        sorted_state_dict = {k: blob[k] for k in sorted(blob.keys())}
        torch.save(sorted_state_dict, buffer)
        serialized_data = buffer.getvalue()
        return serialized_data

    @classmethod
    def to_str(cls, blob: Dict[str, Any]) -> str:
        return f"StateDict({len(blob.keys())})"

    @classmethod
    def _save_operation(cls, blob: Dict[str, Any], path: str) -> None:
        torch.save(blob, path)


class MLDataset(StrEnum):
    CUB = auto()
    FMNIST = auto()
    CelebA = auto()
    CIFAR100 = auto()
    DSPRITES = auto()
    SHAPES = auto()


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if val is None:
        raise EnvironmentError(f"Required environment variable '{name}' is not set.")
    return val


class DB(ABC):
    """
    A simple data management system with deduplicating blob storage.
    """

    root_dir: str

    def __init__(
        self,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
        dry_run: bool = False,
    ):
        """
        Initialize the data management system.

        Args:
            root_dir: Root directory where the database and blob store are located.
            hash_depth: How deep to nest the blob store.
            hash_granularity: Number of bytes to consider for the directory hash.
            dry_run: If enabled, don't store experiments in the database/blob store but print to stdout.
        """
        self.blob_dir = os.path.join(self.root_dir, "blobs")
        for directory in [self.root_dir, self.blob_dir]:
            assert directory is not None
            os.makedirs(directory, exist_ok=True)

        self.hash_depth = hash_depth
        self.hash_granularity = hash_granularity
        self.dry_run = dry_run

    @property
    @abstractmethod
    def db_path(self) -> str:
        """There is no sensible default name for the database, so this should always be overridden in subclasses."""
        pass

    @property
    def output_dir(self) -> str:
        return self.root_dir

    @property
    def output_blob_dir(self) -> str:
        return self.blob_dir

    def query(
        self,
        query: str,
        parameters: Optional[Sequence[Any]] = None,
        schema: Optional[pa.DataFrameModel] = None,
        attach: Dict[str, "DB"] | None = None,
    ) -> pd.DataFrame:
        """
        Execute a query on the database.

        Works the same for pre-trained models, datasets and results, because the csv file is specified in the
        query and the blob paths are specified inside the csv files.

        Args:
            query (str): SQL query to execute.
            parameters: Arguments to pass to the query.
            schema: An optional pandera schema used to validate the retrieved data.

        Returns:
            pd.DataFrame: DataFrame returned by the query.
        """
        assert os.path.exists(os.path.dirname(self.db_path)), self.db_path
        if attach is not None:
            attach_str = "".join(
                f"attach '{db.db_path}' as {name};" for name, db in attach.items()
            )
            query = attach_str + query
        while True:
            try:
                with duckdb.connect(self.db_path) as con:
                    return con.execute(query=query, parameters=parameters).df()
            except duckdb.IOException:
                print(
                    f"Database {self.db_path} seems to be blocked. Retrying in 5 seconds."
                )
                time.sleep(5)

    def query_one(
        self,
        query: str,
        parameters: Optional[Sequence[Any]] = None,
        schema: Optional[pa.DataFrameModel] = None,
    ) -> pd.Series:
        """
        Execute a query that returns a single result on the database.
        """
        records = self.query(query=query, parameters=parameters, schema=schema)
        if len(records) != 1:
            raise ValueError(
                f"{query}, {parameters} returned {len(records)} records when one was expected."
            )
        return records.iloc[0]

    def get_artifact_path_for_checksum(self, checksum: str) -> str:
        blob_bucket = self._get_blob_bucket(checksum)
        blob_path = os.path.join(self.blob_dir, blob_bucket, str(checksum))
        matches = glob.glob(blob_path + "*")
        assert len(matches) != 0, f"{blob_path} does not exist or has been deleted."
        assert len(matches) == 1, f"{blob_path} matches multiple extensions: {matches}"
        return matches[0]

    def load_artifact_for_checksum(self, checksum: str, *args, **kwargs) -> Any:
        path = self.get_artifact_path_for_checksum(checksum)
        _, ext = os.path.splitext(os.path.basename(path))
        blob = load_handlers.get_class(ext).load_from_disk(path, *args, **kwargs)
        return blob

    def load_artifact(self, artifact_name: str, *args, **kwargs) -> Any:
        uri = self._lookup_artifact_uri(artifact_name)
        assert uri is not None, f"Artifact {artifact_name} not found."
        return self.load_artifact_for_checksum(uri, *args, **kwargs)

    def get_artifact_path(self, artifact_name: str) -> Optional[str]:
        uri = self._lookup_artifact_uri(artifact_name)
        assert uri is not None, f"Artifact {artifact_name} not found."
        return self.get_artifact_path_for_checksum(uri)

    def _lookup_artifact_uri(self, artifact_name: str) -> Optional[str]:
        artifact_record = self.query(
            """
            SELECT artifact_uri
            FROM artifacts
            WHERE artifact_name=?
            """,
            [artifact_name],
        )
        if len(artifact_record) == 0:
            return None
        elif len(artifact_record) > 1:
            raise ValueError(
                f"Found multiple artifacts with name {artifact_name}: {artifact_record}"
            )
        uri = artifact_record.iloc[0].loc["artifact_uri"]
        return uri

    def _store_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        ignore_columns: Optional[List[str]] = None,
        replace: bool = False,
    ) -> None:
        """
        Store a DataFrame and optionally associated blobs in the database.
        """
        if ignore_columns is not None:
            df = df.drop(columns=ignore_columns)
        if self.dry_run:
            self._print_table_summary(table_name, df)
            return
        assert os.path.exists(os.path.dirname(self.db_path))
        while True:
            try:
                with duckdb.connect(self.db_path) as con:
                    if replace:
                        con.sql(
                            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df"
                        )
                    else:
                        try:
                            con.sql(
                                f"INSERT INTO {table_name} BY NAME SELECT * FROM df"
                            )
                        except duckdb.CatalogException:
                            con.sql(f"CREATE TABLE {table_name} AS SELECT * FROM df")
                return
            except duckdb.IOException:
                print(
                    f"Database {self.db_path} seems to be blocked. Retrying in 5 seconds."
                )
                time.sleep(5)

    ArtifactTable = TypedDict(
        "ArtifactTable", {"artifact_name": List[str | int], "artifact_uri": List[str]}
    )

    def _store_artifacts(self, artifacts: Dict[int | str, Any]) -> "DB.ArtifactTable":
        artifact_checksums: "DB.ArtifactTable" = {
            "artifact_name": [],
            "artifact_uri": [],
        }
        for key, blob in artifacts.items():
            handler = store_handlers.get_class(type(blob))
            blob = handler.prepare(blob)
            checksum = handler.create_hash(blob)
            blob_bucket = self._get_blob_bucket(checksum)
            global_uri = os.path.join(self.blob_dir, blob_bucket, checksum)
            if not self.dry_run:
                os.makedirs(os.path.dirname(global_uri), exist_ok=True)
                handler.save_to_disk(blob, global_uri)
            artifact_checksums["artifact_name"].append(key)
            artifact_checksums["artifact_uri"].append(checksum)

        if self.dry_run:
            self._print_blob_summary(artifacts)

        return artifact_checksums

    def _get_blob_bucket(self, checksum: str) -> str:
        # Compute the path from the checksum by hashing the first n bytes, e.g., abcdef -> ab/cd/ef
        n_digits = self.hash_granularity * 2  # two hexdigits = 1 byte
        hash_dirs = [
            checksum[i * n_digits : (i + 1) * n_digits] for i in range(self.hash_depth)
        ]
        path = os.path.join(hash_dirs[0], *hash_dirs[1:])
        return path

    def _print_table_summary(self, table_name: str, df: pd.DataFrame) -> None:
        print(f"\n{table_name}")
        print("=" * len(table_name))
        df_str = f"df {df.shape}"
        print(df_str)
        print("-" * len(df_str))
        print(df.head())

    def _print_blob_summary(self, blobs: Dict[int | str, Any]) -> None:
        blobs_str = f"Blobs ({len(blobs)})"
        print("\n" + blobs_str)
        print("-" * len(blobs_str))
        for key, blob in list(blobs.items())[:3]:
            print(f"- {key}: {store_handlers.get_class(type(blob)).to_str(blob)}")
        if len(blobs) > 3:
            print("- ...")


def format_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


class RunDB(DB):
    """
    A database for storing results of experiments.

    Every experiment gets a unique id and its own folder that contains the results.
    Blobs are stored in a deduplicating manner by storing them in a shared blob directory
    and creating hardlinks from the experiment folder.
    """

    run_id: Optional[str]

    def __init__(
        self,
        run_id: Optional[str] = None,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize a new RunDB.

        Args:
            run_id: If provided, the database will be initialized in the given experiment folder (used for analysis).
            hash_depth: The maximum depth of the blob dir.
            hash_granularity: The granularity to use for hashing.
            dry_run: If True, the database will not be initialized, and no blobs will be stored.
        """
        super().__init__(
            hash_depth=hash_depth,
            hash_granularity=hash_granularity,
            dry_run=dry_run,
        )
        if run_id is not None:
            self.run_id = run_id
            self._set_paths(run_id)
        else:
            self.run_id = None
            self._db_path = None
            self._output_dir = None
            self._output_blob_dir = None

    @property
    def db_path(self) -> str:
        if self._db_path is None:
            raise ValueError("Database path not set.")
        return self._db_path

    @db_path.setter
    def db_path(self, value):
        self._db_path = value

    @property
    def output_dir(self) -> str:
        if self._output_dir is None:
            raise ValueError("Output directory not set.")
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value

    @property
    def output_blob_dir(self) -> str:
        if self._output_blob_dir is None:
            raise ValueError("Output blob directory not set.")
        return self._output_blob_dir

    @output_blob_dir.setter
    def output_blob_dir(self, value):
        self._output_blob_dir = value

    def register_run(self, tag: str, args: Optional[argparse.Namespace] = None) -> str:
        """
        Initialize the filesystem for a new run.

        The results from every following store operation will be stored in the same folder.
        """
        if (
            self.run_id is not None
            or self._db_path is not None
            or self._output_dir is not None
            or self._output_blob_dir is not None
        ):
            raise ValueError("Cannot register a new run from analysis mode.")

        # Timezone on the cluster may be different by default, so make sure I set it correctly
        timezone = ZoneInfo("Europe/Berlin")
        while True:
            timestamp = datetime.now(timezone).strftime("%Y-%m-%d-%H:%M:%S")
            run_id = f"exp_{timestamp}_{tag}"
            self._set_paths(run_id)
            if not os.path.exists(self.output_dir) or self.dry_run:
                break
            timeout = random.randint(1, 5)
            print(
                f"Run ID collision ({self.output_dir}). Retrying in {timeout} seconds..."
            )
            time.sleep(timeout)

        # script_args need a run_id field, so that they can be merged with other runs
        # and still be uniquely identified
        script_args = None
        if args is not None:
            script_args = pd.DataFrame([vars(args)])
            script_args["run_id"] = run_id

        self.run_id = run_id

        if self.dry_run:
            print(run_id, timestamp, tag)
            if script_args is not None:
                print(script_args)
            return run_id

        os.makedirs(self.output_dir)
        os.makedirs(self.output_blob_dir)
        if script_args is not None:
            self._store_table("script_args", script_args)

        return run_id

    def store_results(
        self,
        table_name: str,
        df: pd.DataFrame,
        ignore_columns: Optional[List[str]] = None,
        replace: bool = False,
    ):
        self._ensure_registered()
        self._store_table(
            table_name, df, ignore_columns=ignore_columns, replace=replace
        )

    def store_artifacts(self, artifacts: Dict[int | str, Any]) -> None:
        self._ensure_registered()
        artifact_checksums = self._store_artifacts(artifacts)
        if not self.dry_run:
            self._link_artifacts(artifact_checksums)
        artifact_table = pd.DataFrame(artifact_checksums)
        self._store_table("artifacts", artifact_table)

    def clean_blob_store(self):
        """
        Find and delete files with no additional hardlinks (link count = 1).

        Returns:
            Tuple of (deleted_count, total_size)
        """
        deleted_count = 0
        total_size = 0

        # Walk through the directory tree
        for root, _, files in os.walk(self.blob_dir):
            for filename in files:
                filepath = Path(root) / filename

                try:
                    # Get file stats
                    stat_info = filepath.stat()

                    # Check if the file has only 1 hardlink (no additional hardlinks)
                    if stat_info.st_nlink == 1:
                        file_size = stat_info.st_size

                        if self.dry_run:
                            print(
                                f"Would delete: {filepath} ({format_size(file_size)})"
                            )
                        else:
                            print(f"Deleting: {filepath} ({format_size(file_size)})")
                            try:
                                filepath.unlink()
                                deleted_count += 1
                                total_size += file_size
                            except OSError as e:
                                print(f"  Error: Failed to delete {filepath}: {e}")

                except (OSError, PermissionError) as e:
                    print(f"  Error: Cannot access {filepath}: {e}")
                    continue

        return deleted_count, total_size

    def _set_paths(self, run_id: str) -> None:
        self.output_dir = os.path.join(self.root_dir, run_id)
        self.output_blob_dir = os.path.join(self.output_dir, "blobs")
        self.db_path = os.path.join(self.output_dir, "results.db")

    def _ensure_registered(self) -> None:
        if self.run_id is None:
            raise ValueError("Run not registered.")

    def _link_artifacts(self, artifact_checksums: "DB.ArtifactTable") -> None:
        # Additionally, store a hardlink for every blob associated with the experiment.
        # Allows cleaning blobs with link count 1, i.e., those that are no longer referenced by any experiment
        # Adding the file extension is required, because some saving methods, like np.save,
        # add it automatically
        for key, checksum in zip(
            artifact_checksums["artifact_name"], artifact_checksums["artifact_uri"]
        ):
            global_uri = self.get_artifact_path_for_checksum(checksum)
            ext = os.path.splitext(global_uri)[1]
            blob_bucket = self._get_blob_bucket(checksum)
            blob_path = os.path.join(self.output_blob_dir, blob_bucket, checksum)
            local_uri = f"{blob_path}.{ext}"
            os.makedirs(os.path.dirname(local_uri), exist_ok=True)
            os.link(global_uri, local_uri)


class ResultsDB(RunDB):
    root_dir = _require_env("DISCOVERY_RESULTS_ROOT")

    def __init__(
        self,
        run_id: Optional[str] = None,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize a new ResultsDB.

        Args:
            run_id: If provided, the database will be initialized in the given experiment folder (used for analysis).
            hash_depth: The maximum depth of the blob dir.
            hash_granularity: The granularity to use for hashing.
            dry_run: If True, the database will not be initialized, and no blobs will be stored.
        """
        super().__init__(
            run_id=run_id,
            hash_depth=hash_depth,
            hash_granularity=hash_granularity,
            dry_run=dry_run,
        )


class ModelDB(RunDB):
    """
    A database for storing pre-trained models and associated metadata and artifacts.

    There is one key-value store across all models for hyperparameters, and the blob store contains artifacts.
    """

    root_dir = _require_env("DISCOVERY_MODELS_ROOT")

    class Metadata(pa.DataFrameModel):
        tag: str
        dataset: str = pa.Field(isin=[option for option in MLDataset])
        complete_concepts: pd.BooleanDtype = pa.Field(nullable=True)
        architecture_part: str

    class Artifacts(pa.DataFrameModel):
        artifact_name: str
        artifact_uri: str

    def __init__(
        self,
        run_id: Optional[str] = None,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
        dry_run: bool = False,
    ) -> None:
        super().__init__(
            run_id=run_id,
            hash_depth=hash_depth,
            hash_granularity=hash_granularity,
            dry_run=dry_run,
        )

    def store_model(
        self,
        tag: str,
        dataset: MLDataset,
        complete_concepts: Optional[bool],
        architecture_part: str,
        hparams: pd.DataFrame,
        artifacts: Dict[int | str, Any],
        metrics: Optional[pd.DataFrame] = None,
        additional_results: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        """
        Store a pre-trained model and its associated metadata and artifacts.

        Args:
            tag: A unique tag to identify the model.
            dataset: The dataset used to train the model.
            complete_concepts: Whether the model was trained on a complete concept set.
            architecture_part: What part of a larger model is stored (e.g. 'backbone'). Use 'model' when storing the whole model.
            hparams: A DataFrame containing the hyperparameters of the model (i.e., the fields of
            the corresponding config class). This dataframe is assumed to also contain the checksums
            for the state dict and artifacts to find them using queries.
            artifacts: A dict of StorableBlobs containing the artifacts associated with the model.
            metrics: A DataFrame containing measured metrics of the model.
            additional_results: A dictionary of additional results to store.
        """
        self._ensure_registered()

        # Metadata
        metadata = pd.DataFrame(
            {
                "tag": [tag],
                "dataset": [dataset.value],
                "complete_concepts": [complete_concepts],
                "architecture_part": [architecture_part],
            },
            index=[0],
        )
        metadata["complete_concepts"] = metadata["complete_concepts"].astype("boolean")
        self.Metadata.to_schema().validate(metadata)
        self._store_table("metadata", metadata)

        # Artifact table (also used to delete blobs)
        artifact_checksums = self._store_artifacts(artifacts)
        artifact_table = pd.DataFrame(artifact_checksums)
        self.Artifacts.to_schema().validate(artifact_table)
        self._store_table("artifacts", artifact_table)

        self._store_table("hparams", hparams)
        if metrics is not None:
            self._store_table("metrics", metrics)
        if additional_results is not None:
            for table_name, df in additional_results.items():
                self._store_table(table_name, df)

    def add_metric(
        self, metric_name: str, metric_value: int | float | bool | None
    ) -> None:
        self._ensure_registered()
        metrics = self.query("select * from metrics")
        if metric_name in metrics.columns:
            warnings.warn(
                f"Overwriting metric {metric_name}. Previous value: {metrics[metric_name]}, new value: {metric_value}."
            )
        metrics[metric_name] = metric_value
        self._store_table("metrics", metrics, replace=True)

    def load_hparams(self) -> Dict[str, Any]:
        """Load the hyperparameters stored with a model."""
        # Query the hyperparameters for the config
        config_record = self.query_one(query="SELECT * FROM hparams").to_dict()
        return config_record


class CollectionDB:
    db: ResultsDB | ModelDB
    runs: Dict[str, ResultsDB | ModelDB]

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "Use CollectionDB.from_runs or CollectionDB.from_collection instead."
        )

    def _init(
        self,
        collection_id: str,
        run_ids: List[str],
        db_class: Type[ResultsDB | ModelDB],
    ) -> None:
        self.db = db_class(collection_id)
        self.runs = {run_id.split("/")[-1]: db_class(run_id) for run_id in run_ids}

    @classmethod
    def from_runs(cls, run_ids: List[str], db_class: Type[ResultsDB | ModelDB]) -> Self:
        raise NotImplementedError("TODO: Implement collection creation in CollectionDB")
        instance = cls.__new__(cls)
        instance._init(run_ids, db_class)
        return instance

    @classmethod
    def from_collection(
        cls, collection_id: str, db_class: Type[ResultsDB | ModelDB]
    ) -> Self:
        collection_dir = os.path.expandvars(
            os.path.join(db_class.root_dir, collection_id)
        )
        run_ids = [
            os.path.join(collection_id, run_dir)
            for run_dir in os.listdir(collection_dir)
            if run_dir.startswith("exp_")
        ]
        instance = cls.__new__(cls)
        instance._init(collection_id, run_ids, db_class)
        return instance

    def _concat_query(
        self,
        query: str,
        parameters: Optional[Sequence[Any]] = None,
        failure_ok: bool = False,
        attach: Dict[str, "DB"] | None = None,
    ) -> pd.DataFrame:
        dfs = list()
        for run_id, db in self.runs.items():
            try:
                df = db.query(query, parameters, attach=attach)
            except duckdb.BinderException as e:
                if failure_ok:
                    continue
                raise e
            if "run_id" not in df.columns:
                df["run_id"] = run_id.split("/")[-1]
            dfs.append(df)
        assert len(dfs) > 0
        result = pd.concat(dfs).reset_index(drop=True)
        return result

    def aggregate_table(
        self,
        table_name: str,
        query: str,
        parameters: Optional[Sequence[Any]] = None,
        failure_ok: bool = False,
    ) -> None:
        query_result = self._concat_query(query, parameters, failure_ok)
        self.db.store_results(table_name, query_result, replace=True)

    def aggregate_query(
        self,
        query: str,
        parameters: Optional[Sequence[Any]] = None,
        failure_ok: bool = False,
        attach: Dict[str, "DB"] | None = None,
    ) -> pd.DataFrame:
        return self._concat_query(query, parameters, failure_ok, attach)

    def query(
        self, query: str, parameters: Optional[Sequence[Any]] = None
    ) -> pd.DataFrame:
        return self.db.query(query, parameters)


class ResultCollectionDB(CollectionDB):
    @classmethod
    def from_runs(cls, run_ids: List[str]) -> Self:
        return super().from_runs(run_ids, ResultsDB)

    @classmethod
    def from_collection(cls, collection_id: str) -> Self:
        return super().from_collection(collection_id, ResultsDB)


class ModelCollectionDB(CollectionDB):
    @classmethod
    def from_runs(cls, run_ids: List[str]) -> Self:
        return super().from_runs(run_ids, ModelDB)

    @classmethod
    def from_collection(cls, collection_id: str) -> Self:
        return super().from_collection(collection_id, ModelDB)


class DatasetDB(DB):
    """
    A database for storing datasets and associated metadata and artifacts.

    Strict schema enforcement ensures that SQL queries work across different datasets.
    """

    dataset: MLDataset

    class Metadata(pa.DataFrameModel):
        img_id: int = IDField()
        split: str = pa.Field(isin=["train", "val", "test"])

    class ImageClassLabels(pa.DataFrameModel):
        img_id: int = IDField()
        class_id: int = IDField()

    class ImageAttributeLabels(pa.DataFrameModel):
        img_id: int = IDField()
        attribute_id: int = IDField()
        is_present: bool

    class ClassAttributeLabels(pa.DataFrameModel):
        """How often does the class have the attribute across images?"""

        class_id: int
        attribute_id: int
        confidence: float  # "confidence" for consistency with CUB

    class Images(pa.DataFrameModel):
        img_id: int = IDField()
        image: str

    class Mutexes(pa.DataFrameModel):
        group_id: int = IDField()
        attribute_id: int = IDField(unique=True)

    class ConceptDAG(pa.DataFrameModel):
        scenario: str
        source_id: int = IDField()
        target_id: int = IDField()
        is_connected: bool

    class TaskDAG(pa.DataFrameModel):
        scenario: str
        source_id: int = IDField()
        target_id: int = IDField()
        is_connected: bool

    def __init__(
        self,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
        dry_run: bool = False,
    ) -> None:
        super().__init__(
            hash_depth=hash_depth,
            hash_granularity=hash_granularity,
            dry_run=dry_run,
        )
        self.dataset_name: str = str(self.dataset)

    @property
    def db_path(self) -> str:
        return os.path.join(self.root_dir, "annotations.db")

    @property
    def metadata_schema(self) -> pa.DataFrameSchema:
        return self.Metadata.to_schema()

    @property
    def image_class_labels_schema(self) -> pa.DataFrameSchema:
        return self.ImageClassLabels.to_schema()

    @property
    def image_attribute_labels_schema(self) -> pa.DataFrameSchema:
        return self.ImageAttributeLabels.to_schema()

    @property
    def class_attribute_labels_schema(self) -> pa.DataFrameSchema:
        return self.ClassAttributeLabels.to_schema()

    @property
    def images_schema(self) -> pa.DataFrameSchema:
        return self.Images.to_schema()

    @property
    def mutexes_schema(self) -> pa.DataFrameSchema:
        return self.Mutexes.to_schema()

    @property
    def concept_dag_schema(self) -> pa.DataFrameSchema:
        return self.ConceptDAG.to_schema()

    @property
    def task_dag_schema(self) -> pa.DataFrameSchema:
        return self.TaskDAG.to_schema()

    @property
    @abstractmethod
    def class_names_schema(self) -> pa.DataFrameSchema:
        raise NotImplementedError

    @property
    @abstractmethod
    def attribute_names_schema(self) -> pa.DataFrameSchema:
        raise NotImplementedError

    def store_metadata(self, df: pd.DataFrame) -> None:
        self.metadata_schema.validate(df)
        self._store_table("metadata", df)

    def store_image_class_labels(self, df: pd.DataFrame) -> None:
        self.image_class_labels_schema.validate(df)
        self._store_table("image_class_labels", df)

    def store_image_attribute_labels(self, df: pd.DataFrame) -> None:
        self.image_attribute_labels_schema.validate(df)
        self._store_table("image_attribute_labels", df)

    def store_class_attribute_labels(self, df: pd.DataFrame) -> None:
        self.class_attribute_labels_schema.validate(df)
        self._store_table("class_attribute_labels", df)

    def store_images(self, images: Dict[int | str, PILImage.Image | str]) -> None:
        images = {int(k): v for k, v in images.items()}
        artifact_checksums = self._store_artifacts(images)
        artifact_table = pd.DataFrame(artifact_checksums)
        artifact_table = artifact_table.rename(
            columns={"artifact_name": "img_id", "artifact_uri": "image"}
        )
        self.images_schema.validate(artifact_table)
        self._store_table("images", artifact_table)

    def store_class_names(self, df: pd.DataFrame) -> None:
        self.class_names_schema.validate(df)
        self._store_table("class_names", df)

    def store_attribute_names(self, df: pd.DataFrame) -> None:
        self.attribute_names_schema.validate(df)
        self._store_table("attribute_names", df)

    def store_mutexes(self, df: pd.DataFrame) -> None:
        self.mutexes_schema.validate(df)
        self._store_table("mutexes", df)

    def store_concept_dag(self, df: pd.DataFrame) -> None:
        self.concept_dag_schema.validate(df)
        self._store_table("concept_dag", df)

    def store_task_dag(self, df: pd.DataFrame) -> None:
        self.task_dag_schema.validate(df)
        self._store_table("task_dag", df)

    def _lookup_artifact_uri(self, artifact_name: str) -> Optional[str]:
        image_record = self.query(
            """
            SELECT image
            FROM images
            WHERE img_id=?
            """,
            [artifact_name],
        )
        assert len(image_record) == 1, (
            f"Found {len(image_record)} records for {artifact_name}"
        )
        uri = image_record.iloc[0].loc["image"]
        return uri


dataset_db_registry = Registry[DatasetDB, MLDataset]("DatasetRegistry")


@dataset_db_registry.register(MLDataset.CUB)
class CUBDB(DatasetDB):
    dataset: MLDataset = MLDataset.CUB
    root_dir = os.path.join(_require_env("DISCOVERY_DATA_ROOT"), str(MLDataset.CUB))

    class ClassNames(pa.DataFrameModel):
        class_id: int = IDField(le=200)

    class AttributeNames(pa.DataFrameModel):
        attribute_id: int = IDField(le=112)

    def __init__(
        self,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
        dry_run: bool = False,
    ) -> None:
        super().__init__(
            hash_depth=hash_depth,
            hash_granularity=hash_granularity,
            dry_run=dry_run,
        )

    @property
    def class_names_schema(self) -> pa.DataFrameSchema:
        return self.ClassNames.to_schema()

    @property
    def attribute_names_schema(self) -> pa.DataFrameSchema:
        return self.AttributeNames.to_schema()


@dataset_db_registry.register(MLDataset.FMNIST)
class FMNISTDB(DatasetDB):
    dataset: MLDataset = MLDataset.FMNIST
    root_dir = os.path.join(_require_env("DISCOVERY_DATA_ROOT"), str(MLDataset.FMNIST))

    class ClassNames(pa.DataFrameModel):
        class_id: int = IDField(le=10)

    class AttributeNames(pa.DataFrameModel):
        attribute_id: int = IDField(le=11)

    def __init__(
        self,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
        dry_run: bool = False,
    ) -> None:
        super().__init__(
            hash_depth=hash_depth,
            hash_granularity=hash_granularity,
            dry_run=dry_run,
        )

    @property
    def class_names_schema(self) -> pa.DataFrameSchema:
        return self.ClassNames.to_schema()

    @property
    def attribute_names_schema(self) -> pa.DataFrameSchema:
        return self.AttributeNames.to_schema()


@dataset_db_registry.register(MLDataset.CelebA)
class CelebADB(DatasetDB):
    dataset: MLDataset = MLDataset.CelebA
    root_dir = os.path.join(_require_env("DISCOVERY_DATA_ROOT"), str(MLDataset.CelebA))

    class AttributeNames(pa.DataFrameModel):
        attribute_id: int = IDField(le=40)

    def __init__(
        self,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
        dry_run: bool = False,
    ) -> None:
        super().__init__(
            hash_depth=hash_depth,
            hash_granularity=hash_granularity,
            dry_run=dry_run,
        )

    @property
    def class_names_schema(self) -> pa.DataFrameSchema:
        raise NotImplementedError

    @property
    def attribute_names_schema(self) -> pa.DataFrameSchema:
        return self.AttributeNames.to_schema()


@dataset_db_registry.register(MLDataset.CIFAR100)
class CIFAR100DB(DatasetDB):
    dataset: MLDataset = MLDataset.CIFAR100
    root_dir = os.path.join(
        _require_env("DISCOVERY_DATA_ROOT"), str(MLDataset.CIFAR100)
    )

    class ClassNames(pa.DataFrameModel):
        class_id: int = IDField(le=100)

    class AttributeNames(pa.DataFrameModel):
        attribute_id: int = IDField(le=20)

    def __init__(
        self,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
        dry_run: bool = False,
    ) -> None:
        super().__init__(
            hash_depth=hash_depth,
            hash_granularity=hash_granularity,
            dry_run=dry_run,
        )

    @property
    def class_names_schema(self) -> pa.DataFrameSchema:
        return self.ClassNames.to_schema()

    @property
    def attribute_names_schema(self) -> pa.DataFrameSchema:
        return self.AttributeNames.to_schema()


@dataset_db_registry.register(MLDataset.DSPRITES)
class DSPRITESDB(DatasetDB):
    dataset: MLDataset = MLDataset.DSPRITES
    root_dir = os.path.join(
        _require_env("DISCOVERY_DATA_ROOT"), str(MLDataset.DSPRITES)
    )

    class ClassNames(pa.DataFrameModel):
        class_id: int = IDField(le=2)

    class AttributeNames(pa.DataFrameModel):
        attribute_id: int = IDField(le=11)

    def __init__(
        self,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
        dry_run: bool = False,
    ) -> None:
        super().__init__(
            hash_depth=hash_depth,
            hash_granularity=hash_granularity,
            dry_run=dry_run,
        )

    @property
    def class_names_schema(self) -> pa.DataFrameSchema:
        return self.ClassNames.to_schema()

    @property
    def attribute_names_schema(self) -> pa.DataFrameSchema:
        return self.AttributeNames.to_schema()


@dataset_db_registry.register(MLDataset.SHAPES)
class SHAPESDB(DatasetDB):
    dataset: MLDataset = MLDataset.SHAPES
    root_dir = os.path.join(_require_env("DISCOVERY_DATA_ROOT"), str(MLDataset.SHAPES))

    class ClassNames(pa.DataFrameModel):
        class_id: int = IDField(le=7)
        class_name: str

    class AttributeNames(pa.DataFrameModel):
        attribute_id: int = IDField(le=15)
        attribute_name: str

    def __init__(
        self,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
        dry_run: bool = False,
    ) -> None:
        super().__init__(
            hash_depth=hash_depth,
            hash_granularity=hash_granularity,
            dry_run=dry_run,
        )

    @property
    def class_names_schema(self) -> pa.DataFrameSchema:
        return self.ClassNames.to_schema()

    @property
    def attribute_names_schema(self) -> pa.DataFrameSchema:
        return self.AttributeNames.to_schema()


def _resolve_concept_names_to_ids(db: DatasetDB, concepts: List[str | int]) -> set[int]:
    concept_ids: Set[int] = set()
    for concept in concepts:
        if type(concept) is str:
            concept = db.query_one(
                """
                SELECT attribute_id
                FROM attribute_names
                WHERE attribute_name=?
                """,
                [concept],
            ).item()
        assert isinstance(concept, int), f"Concept {concept} is not an integer"
        concept_ids.add(concept)
    return concept_ids


def get_cream_dataset(
    dataset_type: MLDataset,
    split: Literal["train", "val", "test"],
    sort: bool = True,
    enable_caching: bool = True,
    transform: Optional[Callable] = None,
    drop_concepts: Literal["all"] | List[int | str] | None = None,
    n_samples: int | None = None,
    **kwargs,
) -> Dataset:
    """Factory function to create PyTorch compatible datasets"""
    # Load dataset from the database
    db = dataset_db_registry.create(dataset_type, **kwargs)
    n_total = db.query_one(
        query="SELECT count(*) AS n_total FROM metadata m WHERE m.split=?",
        parameters=[split],
    )["n_total"]
    query = f"""
WITH filtered_attrs AS
  (SELECT *
   FROM image_attribute_labels icl
   WHERE EXISTS
       (SELECT *
        FROM concept_dag cd
        WHERE cd.source_id = icl.attribute_id)),
     attrs AS (PIVOT filtered_attrs ON 'concept_' || attribute_id USING first(is_present)),
     dataset AS
  (SELECT i.img_id,
          i.image,
          icl.class_id - 1 AS label,
          columns('concept_*')
   FROM images i
   JOIN image_class_labels icl ON icl.img_id = i.img_id
   JOIN metadata m ON m.img_id = i.img_id
   JOIN attrs ON attrs.img_id = i.img_id
   WHERE m.split = ?
   ORDER BY {"i.img_id" if sort else "hash(i.img_id, 42)"}),
     dataset_sample AS
  (SELECT percent_rank() OVER (PARTITION BY label
                               ORDER BY hash(img_id, 42)) AS sample_percent_rank,
          *
   FROM dataset)
    """

    # Implement stratified sampling if n_samples is specified
    if n_samples is not None:
        query += """
        SELECT columns(* exclude sample_percent_rank) FROM dataset_sample WHERE sample_percent_rank < ?
        """
        query_result = db.query(query=query, parameters=[split, n_samples / n_total])
    else:
        query += """
        SELECT * FROM dataset
        """
        query_result = db.query(query=query, parameters=[split])

    # Need to sort the concepts by id so that the one-hot vectors are correct
    # Need to do this in pandas, because duckdb doesn't support reordering columns and always
    # sorts lexicographically, i.e. 1, 10, 11, 2, 3, ...
    concept_columns = []
    other_columns = []
    for column in query_result.columns:
        if column.startswith("concept_"):
            concept_columns.append(column)
        else:
            other_columns.append(column)

    if drop_concepts == "all":
        query_result = query_result[other_columns]
    else:
        if type(drop_concepts) is list:
            drop_concept_ids = _resolve_concept_names_to_ids(db, drop_concepts)
            drop_column_names = ["concept_" + str(i) for i in drop_concept_ids]
            concept_columns = list(set(concept_columns) - set(drop_column_names))
        concept_columns = sorted(concept_columns, key=lambda x: int(x.split("_")[-1]))
        sorted_columns = other_columns + concept_columns
        query_result[concept_columns] = query_result[concept_columns].astype(float)
        query_result = query_result[sorted_columns]
        query_result["concept_labels"] = query_result[concept_columns].apply(  # type: ignore
            list, axis=1
        )
        query_result = query_result.drop(columns=concept_columns)
    assert isinstance(query_result, pd.DataFrame), type(query_result)

    # Initialize the hf dataset
    if enable_caching:
        # Cache the query result to a file to enable huggingface caching
        cache_root = os.getenv("DISCOVERY_CACHE_ROOT")
        assert cache_root is not None, "Cache root directory must exist"
        query_result_hash = hashlib.sha256(pickle.dumps(query_result)).hexdigest()[:16]
        cache_path = os.path.join(cache_root, f"dataset_{query_result_hash}")
        if not os.path.exists(cache_path):
            ds = Dataset.from_pandas(query_result)  # type: ignore
            ds.save_to_disk(cache_path)
        dataset = load_from_disk(cache_path)
    else:
        dataset = Dataset.from_pandas(query_result)  # type: ignore

    # Load images
    dataset = dataset.map(
        lambda x: {"image": db.get_artifact_path_for_checksum(x["image"])},
        desc="Loading images",
        num_proc=get_num_cpu_workers(),
    ).cast_column("image", Image())

    # Add transform if specified
    if transform is not None:
        dataset.set_transform(
            lambda xs: {"image": torch.stack([transform(x) for x in xs["image"]])},
            columns=["image"],
            output_all_columns=True,
        )

    assert isinstance(dataset, Dataset), dataset

    return dataset


def get_cream_annotations(
    dataset_type: MLDataset, scenario: str, drop_concepts: List[str | int] | None = None
) -> Tuple[Dict[str, pd.DataFrame], dict[str, dict[int, int]]]:
    # Load database for the dataset
    db = dataset_db_registry.create(dataset_type)

    # Load dataframes with the annotations
    mutexes = db.query(
        """
        SELECT group_id AS group_id, attribute_id AS attribute_id
        FROM mutexes
        """
    )
    concept_dag = db.query(
        """
        SELECT source_id AS source_id, target_id AS target_id, is_connected
        FROM concept_dag
        WHERE scenario=?
        """,
        parameters=[scenario],
    )
    task_dag = db.query(
        """
        SELECT source_id AS source_id, target_id AS target_id, is_connected
        FROM task_dag
        WHERE scenario=?
        """,
        parameters=[scenario],
    )
    if len(concept_dag) == 0:
        raise ValueError(f"No concept_dag found for scenario {scenario}")
    if len(task_dag) == 0:
        raise ValueError(f"No task_dag found for scenario {scenario}")

    # Sanity checks
    if not mutexes.empty:
        assert (mutexes["group_id"].min() >= 0) and (mutexes["attribute_id"].min() >= 0)
    assert (concept_dag["source_id"].min() >= 0) and (
        concept_dag["target_id"].min() >= 0
    )
    assert (task_dag["source_id"].min() >= 0) and (task_dag["target_id"].min() >= 0)

    # Drop concepts if specified
    if drop_concepts is not None:
        drop_concept_ids = _resolve_concept_names_to_ids(db, drop_concepts)
        drop_concept_ids = {v for v in drop_concept_ids}
        mutexes = mutexes.merge(
            mutexes.groupby("group_id")
            .agg(lambda xs: sum([x in drop_concept_ids for x in xs]))  # > 0 -> overlap
            .reset_index()
            .loc[lambda x: x["attribute_id"] == 0, ["group_id"]],  # == 0 -> no overlap
            how="right",
            on="group_id",
        )
        concept_dag = concept_dag[
            (~concept_dag["source_id"].isin(drop_concept_ids))  # type: ignore
            & (~concept_dag["target_id"].isin(drop_concept_ids))  # type: ignore
        ]
        task_dag = task_dag[~task_dag["source_id"].isin(drop_concept_ids)]  # type: ignore

    # Remap attribute IDs to contiguous range [0, num_nodes - 1]
    attribute_ids = concept_dag["source_id"].unique()  # type: ignore
    aid2idx = {old_id: new_id for new_id, old_id in enumerate(sorted(attribute_ids))}
    concept_dag["source_id"] = concept_dag["source_id"].map(aid2idx)  # type: ignore
    concept_dag["target_id"] = concept_dag["target_id"].map(aid2idx)  # type: ignore
    task_dag["source_id"] = task_dag["source_id"].map(aid2idx)  # type: ignore
    mutexes["attribute_id"] = mutexes["attribute_id"].map(aid2idx)  # type: ignore
    assert not mutexes["attribute_id"].isna().any(), mutexes  # type: ignore
    mutex_attribute_ids = set(concept_dag["target_id"].unique())  # type: ignore
    dag_attribute_ids = set(concept_dag["target_id"].unique())  # type: ignore
    assert mutex_attribute_ids == dag_attribute_ids

    # Remap class IDs to contiguous range [0, num_classes - 1]
    # For binary class labels, only one class (the positive one) may appear in the dag
    class_ids = task_dag["target_id"].unique()  # type: ignore
    cid2idx = {old_id: old_id - 1 for old_id in sorted(class_ids)}
    task_dag["target_id"] = task_dag["target_id"].map(cid2idx)  # type: ignore

    # Remap mutex group IDs to contiguous range
    group_ids = mutexes["group_id"].unique()
    gid2idx = {old_id: new_id for new_id, old_id in enumerate(sorted(group_ids))}
    mutexes["group_id"] = mutexes["group_id"].map(gid2idx)  # type: ignore

    return {"mutexes": mutexes, "concept_dag": concept_dag, "task_dag": task_dag}, {
        "attribute_id_mapping": {v: k for k, v in aid2idx.items()},
        "class_id_mapping": {v: k for k, v in cid2idx.items()},
        "group_id_mapping": {v: k for k, v in gid2idx.items()},
    }  # type: ignore
