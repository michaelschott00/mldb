from abc import ABC, abstractmethod
import hashlib
import io
import os
import re
import shutil
import warnings
from datetime import datetime
import numpy as np
from PIL import Image as PILImage
from os import environ
from pathlib import Path
from typing import Any, Self
from collections.abc import Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import torch
import sqlite3

from db.registry import Registry, TypeHandlerRegistry

DEFAULT_HASH_DEPTH = 3
DEFAULT_HASH_GRANULARITY = 1


def _require_env(name: str) -> str:
    val = environ.get(name)
    if val is None:
        raise EnvironmentError(f"Required environment variable '{name}' is not set.")
    return val

class StorableBlob(ABC):
    # TODO: replace Any with type parameter
    ext: str = ""

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
        # TODO: Replace os.path with pathlib.Path
        name, ext = os.path.splitext(path)
        if cls.ext != "":
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
    ext: str = ".npy"

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


@store_handlers.register(pd.DataFrame)
@load_handlers.register(".csv")
class TableBlob(StorableBlob):
    ext: str = ".csv"

    @classmethod
    def to_bytes(cls, blob: pd.DataFrame | pd.Series) -> bytes:
        return blob.to_csv(None).encode('utf-8')

    @classmethod
    def load_from_disk(cls, path: str, *args, **kwargs) -> pd.DataFrame:
        df = pd.read_csv(path, *args, **kwargs)
        assert isinstance(df, pd.DataFrame), type(df)
        return df

    @classmethod
    def to_str(cls, blob: pd.DataFrame | pd.Series) -> str:
        return f"Table({blob.shape})"

    @classmethod
    def _save_operation(cls, blob: pd.DataFrame | pd.Series, path: str) -> None:
        blob.to_csv(path, index=False)


@store_handlers.register(PILImage.Image)
@load_handlers.register(".jpg")
@load_handlers.register(".png")
class ImageBlob(StorableBlob):
    ext: str = ".png"

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


# TODO: Lazy import torch
@store_handlers.register(torch.Tensor)
@load_handlers.register(".pth")
class TensorBlob(StorableBlob):
    ext: str = ".pth"

    @classmethod
    def to_bytes(cls, blob: torch.Tensor) -> bytes:
        return blob.cpu().numpy().tobytes()

    @classmethod
    def load_from_disk(cls, path: str, *args, **kwargs) -> torch.Tensor:
        return torch.load(path, *args, **kwargs)

    @classmethod
    def to_str(cls, blob: torch.Tensor) -> str:
        return f"Tensor({list(blob.shape)}, {str(blob.dtype)})"

    @classmethod
    def _save_operation(cls, blob: torch.Tensor, path: str) -> None:
        torch.save(blob.cpu(), path)


# Needed despite the TensorBlob because hashes are computed differently.
@store_handlers.register(dict)
@load_handlers.register(".state_dict")
class TorchStateDictBlob(StorableBlob):
    ext = ".state_dict"

    @classmethod
    def load_from_disk(cls, path: str, *args, **kwargs) -> dict[str, Any]:
        return torch.load(path, *args, **kwargs)

    @classmethod
    def to_bytes(cls, blob: dict[str, Any]) -> bytes:
        blob = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in blob.items()}
        buffer = io.BytesIO()
        sorted_state_dict = {k: blob[k] for k in sorted(blob.keys())}
        torch.save(sorted_state_dict, buffer)
        serialized_data = buffer.getvalue()
        return serialized_data

    @classmethod
    def to_str(cls, blob: dict[str, Any]) -> str:
        return f"StateDict({len(blob.keys())})"

    @classmethod
    def _save_operation(cls, blob: dict[str, Any], path: str) -> None:
        blob = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in blob.items()}
        torch.save(blob, path)



class ExperimentStore:

    def __init__(
        self,
        root_dir: str,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
    ) -> None:
        self._root_dir = Path(root_dir)
        self._db_path: Path = self._root_dir / "index.db"
        self._hash_depth = hash_depth
        self._hash_granularity = hash_granularity
        self._run_id = None
        self._run_dir = None
        self._root_dir.mkdir(exist_ok=True, parents=True)
        self._run_sql(
            """
            create table if not exists runs (
                run_id VARCHAR PRIMARY KEY,
                run_name VARCHAR,
                run_timestamp TIMESTAMP
            )
            """
        )
        self._run_sql(
            """
            create table if not exists artifacts (
                run_id VARCHAR,
                artifact_name VARCHAR,
                artifact_checksum VARCHAR,
                PRIMARY KEY (run_id, artifact_name)
            )
            """
        )
        self._runs_table = "runs"
        self._artifacts_table = "artifacts"

    @classmethod
    def from_env(
        cls,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
    ) -> Self:
        root_dir = _require_env("DATA_ROOT")
        return cls(
            root_dir=root_dir,
            hash_depth=hash_depth,
            hash_granularity=hash_granularity,
        )

    @classmethod
    def from_run(cls, run_id: str, root_dir: str | None = None) -> Self:
        if root_dir is None:
            root_dir = _require_env("DATA_ROOT")
        # TODO: Detect hash depth and hash granularity
        instance = cls(root_dir=root_dir)
        instance._run_id = run_id
        instance._run_dir = instance._root_dir / instance._run_id
        return instance

    def query(self, query: str, parameters: Sequence[Any] | None = None) -> pd.DataFrame:
        data = self._run_sql(query, parameters)
        df = pd.DataFrame(data)
        return df
    
    def query_one(self, query: str, parameters: Sequence[Any] | None = None) -> pd.Series:
        df = self.query(query, parameters)
        return df.iloc[0]

    def query_value(self, query: str, parameters: Sequence[Any] | None = None) -> Any:
        df = self.query(query, parameters)
        return df.iloc[0, 0]

    def register_run(self, name: str) -> str:
        timezone = ZoneInfo("Europe/Berlin")
        timestamp = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
        self._run_id = f"exp_{timestamp.replace(' ', '-')}_{name}"
        self._run_dir = self._root_dir / self._run_id
        self._run_dir.mkdir()
        self._run_sql(f"insert into {self._runs_table} values (?, ?, ?)", [self._run_id, name, timestamp])
        return self._run_id

    def store(
        self,
        artifacts: dict[int | str, Any]
    ) -> None:
        self._check_registered()
        artifact_checksums: Sequence[tuple[str, str, str]] = list()
        for key, blob in artifacts.items():
            checksum = self._get_checksum(blob)
            uri = self._get_uri_for_checksum(checksum)
            self._save_to_disk(blob, uri)
            artifact_checksums.append((self._run_id, str(key), checksum))
        self._run_sql_many(f"insert into {self._artifacts_table} values (?, ?, ?)", artifact_checksums)

    def load(self, run_id: str, artifact_name: str) -> Any:
        checksum = self.query_value(
            f"""
            select artifact_checksum
            from {self._artifacts_table}
            where run_id=? and artifact_name=?
            """,
            parameters=[run_id, artifact_name]
        )
        uri_no_ext = Path(self._get_uri_for_checksum(checksum))
        uri_matches = list(uri_no_ext.parent.glob(f"{uri_no_ext.stem}*"))
        if len(uri_matches) == 0:
            raise ValueError(f"No match for artifact {artifact_name}")
        if len(uri_matches) > 1:
            raise ValueError(f"Multiple matches for artifact {artifact_name}")
        blob = self._load_from_disk(str(uri_matches[0]))
        return blob

    def _check_registered(self) -> None:
        if self._run_dir is None or self._run_id is None:
            raise ValueError("Register run first")

    def _get_checksum(self, blob: Any) -> str:
        return store_handlers.get_class(type(blob)).create_hash(blob)
    
    def _get_blob_bucket(self, checksum: str) -> list[str]:
        n_digits = self._hash_granularity * 2  # two hexdigits = 1 byte
        blob_bucket = [
            checksum[i * n_digits : (i + 1) * n_digits] for i in range(self._hash_depth)
        ]
        return blob_bucket
    
    def _get_uri_for_checksum(self, checksum: str) -> str:
        self._check_registered()
        blob_bucket = self._get_blob_bucket(checksum)
        blob_bucket_dir = self._run_dir / Path(*blob_bucket)
        blob_bucket_dir.mkdir(exist_ok=True, parents=True)
        uri = blob_bucket_dir / checksum
        return str(uri)

    def _save_to_disk(self, blob: Any, uri: str) -> None:
        store_handlers.get_class(type(blob)).save_to_disk(blob, str(uri))

    def _load_from_disk(self, uri: str) -> None:
        _, ext = os.path.splitext(os.path.basename(uri))
        blob = load_handlers.get_class(ext).load_from_disk(uri)
        return blob

    def _run_sql(self, query: str, parameters: Sequence[Any] | None = None) -> list[Any]:
        if parameters is None:
            parameters = list()
        with sqlite3.connect(self._db_path) as con:
            return con.execute(query, parameters).fetchall()

    def _run_sql_many(self, query: str, parameters: Sequence[Any]) -> list[Any]:
        with sqlite3.connect(self._db_path) as con:
            return con.executemany(query, parameters).fetchall()

