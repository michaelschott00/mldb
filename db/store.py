import hashlib
import io
import os
import re
import shutil
import sqlite3
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from os import environ
from pathlib import Path
from typing import Any, Self
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage

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
        return blob.to_csv(None).encode("utf-8")

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


# class RunLogger:
#
#     def __init__(self) -> None:
#         pass
#
#     def register_run(self, name: str) -> str:
#         timezone = ZoneInfo("Europe/Berlin")
#         timestamp = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
#         run_id = f"run_{timestamp.replace(' ', '-')}_{name}"
#         self._sql_execute(f"insert into {self._runs_table} values (?, ?, ?)", [run_id, name, timestamp])
#         return run_id


class RunStore:
    def __init__(
        self,
        root_dir: str,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
    ) -> None:
        self._hash_depth = hash_depth
        self._hash_granularity = hash_granularity

        self._root_dir = Path(root_dir)
        self._root_dir.mkdir(exist_ok=True, parents=True)
        self._db_path: Path = self._root_dir / "index.db"

        self._artifacts_table = "artifacts"
        self._execute_sql(
            f"""
            create table if not exists {self._artifacts_table} (
                run_id VARCHAR,
                artifact_name VARCHAR,
                artifact_checksum VARCHAR,
                PRIMARY KEY (run_id, artifact_name)
            )
            """
        )
        self._runs_table = "runs"
        self._execute_sql(
            f"""
            create table if not exists {self._runs_table} (
                run_id VARCHAR PRIMARY KEY,
                run_name VARCHAR,
                run_timestamp TIMESTAMP
            )
            """
        )
        self._tags_table = "tags"
        self._execute_sql(
            f"""
            create table if not exists {self._tags_table} (
                run_id VARCHAR,
                tag VARCHAR,
                PRIMARY KEY (run_id, tag)
            )
            """
        )

    @classmethod
    def from_env(cls) -> Self:
        root_dir = _require_env("DATA_ROOT")
        return cls(root_dir=root_dir)

    def query(
        self, query: str, parameters: Sequence[Any] | None = None
    ) -> pd.DataFrame:
        data = self._query_rows(query, parameters)
        df = pd.DataFrame(data)
        return df

    def query_run(self, include_tags: list[str], exclude_tags: list[str]) -> list[str]:
        query = self._get_tag_query(include_tags, exclude_tags)
        matching_runs = self._query_rows(query, include_tags + exclude_tags)
        return matching_runs["run_id"]

    def create_run(self, name: str, tags: list[str] | None = None) -> str:
        timezone = ZoneInfo("Europe/Berlin")
        timestamp = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
        run_id = f"run_{timestamp.replace(' ', '-')}_{uuid.uuid4().hex[:8]}_{name}"
        self._execute_sql(
            f"insert into {self._runs_table} values (?, ?, ?)",
            [run_id, name, timestamp],
        )
        if tags is not None:
            self.add_tags(run_id, tags)
        return run_id

    def add_tags(self, run_id: str, tags: list[str]) -> None:
        self._executemany_sql(
            f"insert into {self._tags_table} values (?, ?)",
            [(run_id, tag) for tag in tags],
        )

    def store(self, run_id: str, artifacts: dict[int | str, Any]) -> None:
        artifact_checksums: Sequence[tuple[str, str, str]] = list()
        for key, blob in artifacts.items():
            checksum = self._get_checksum(blob)
            uri = self._get_uri_for_checksum(checksum)
            self._save_to_disk(blob, uri)
            artifact_checksums.append((run_id, str(key), checksum))
        self._executemany_sql(
            f"insert into {self._artifacts_table} values (?, ?, ?)", artifact_checksums
        )

    def load(self, run_id: str, artifact_name: str) -> Any:
        checksum = self._query_value(
            f"""
            select artifact_checksum
            from {self._artifacts_table}
            where run_id=? and artifact_name=?
            """,
            parameters=[run_id, artifact_name],
        )
        blob = self._get_blob_for_checksum(checksum)
        return blob

    def load_by_tags(
        self, artifact_name: str, include_tags: list[str], exclude_tags: list[str]
    ) -> dict[str, Any]:
        query_result = self._do_tag_search(artifact_name, include_tags, exclude_tags)
        result = dict()
        for run_id, checksum in zip(
            query_result["run_id"], query_result["artifact_checksum"]
        ):
            assert run_id not in result, result
            blob = self._get_blob_for_checksum(checksum)
            result[run_id] = blob
        return result

    def load_duckdb(
        self,
        artifact_names: list[str],
        include_tags: list[str],
        exclude_tags: list[str],
    ) -> duckdb.DuckDBPyConnection:
        query_result = self._do_tag_search(artifact_name, include_tags, exclude_tags)
        con = duckdb.connect()
        for run_id, checksum in zip(
            query_result["run_id"], query_result["artifact_checksum"]
        ):
            uri = self._get_uri_for_checksum(checksum, resolve_ext=True)
            assert uri.endswith(".csv")
            # TODO: Dangerous string replacement
            con.sql(f"create table {artifact_name} as from '{uri}'")
        return con

    def _get_tag_query(self, include_tags: list[str], exclude_tags: list[str]) -> str:
        query = ""
        for i in range(len(include_tags) + len(exclude_tags)):
            query += """
                select run_id
                from tags
                where tag=?
            """
            if i < len(include_tags) - 1:
                query += "intersect"
            elif i < len(include_tags) + len(exclude_tags) - 1:
                query += "except"
        return query

    def _do_tag_search(
        self, artifact_name: str, include_tags: list[str], exclude_tags: list[str]
    ) -> dict[str, Any]:
        tags_query = self._get_tag_query(include_tags, exclude_tags)
        # TODO: Use that selection join (semi-join?)
        query = f"""
            with matching_runs as ({tags_query})
            select a.run_id, artifact_checksum
            from matching_runs r
            join {self._artifacts_table} a
            on r.run_id=a.run_id
            where a.artifact_name=?
        """
        query_result = self._query_rows(
            query, include_tags + exclude_tags + [artifact_name]
        )
        return query_result

    def _get_checksum(self, blob: Any) -> str:
        return store_handlers.get_class(type(blob)).create_hash(blob)

    def _get_blob_bucket(self, checksum: str) -> list[str]:
        n_digits = self._hash_granularity * 2  # two hexdigits = 1 byte
        blob_bucket = [
            checksum[i * n_digits : (i + 1) * n_digits] for i in range(self._hash_depth)
        ]
        return blob_bucket

    def _get_uri_for_checksum(self, checksum: str, resolve_ext: bool = False) -> str:
        blob_bucket = self._get_blob_bucket(checksum)
        blob_bucket_dir = self._root_dir / Path(*blob_bucket)
        blob_bucket_dir.mkdir(exist_ok=True, parents=True)
        uri_no_ext = blob_bucket_dir / checksum
        if resolve_ext:
            uri_matches = list(blob_bucket_dir.glob(checksum + "*"))
            if len(uri_matches) == 0:
                raise ValueError(f"No match.")
            if len(uri_matches) > 1:
                raise ValueError(f"Multiple matches.")
            return str(uri_matches[0])
        return str(uri_no_ext)

    def _get_blob_for_checksum(self, checksum: str) -> Any:
        uri = self._get_uri_for_checksum(checksum, resolve_ext=True)
        blob = self._load_from_disk(uri)
        return blob

    def _save_to_disk(self, blob: Any, uri: str) -> None:
        store_handlers.get_class(type(blob)).save_to_disk(blob, str(uri))

    def _load_from_disk(self, uri: str) -> None:
        _, ext = os.path.splitext(os.path.basename(uri))
        blob = load_handlers.get_class(ext).load_from_disk(uri)
        return blob

    def _execute_sql(self, query: str, parameters: Sequence[Any] | None = None) -> None:
        with sqlite3.connect(self._db_path) as con:
            con.execute(query) if parameters is None else con.execute(query, parameters)

    def _executemany_sql(self, query: str, parameters: Sequence[Any]) -> None:
        with sqlite3.connect(self._db_path) as con:
            con.executemany(query, parameters)

    def _query_result_to_dict(
        self, column_names: list[str], rows: list[tuple[Any, ...]]
    ) -> dict[str, list[Any]]:
        result = dict()
        for column_name, column_data in zip(column_names, zip(*rows)):
            result[column_name] = column_data
        return result

    def _query_value(self, query: str, parameters: Sequence[Any] | None = None) -> Any:
        with sqlite3.connect(self._db_path) as con:
            cursor = (
                con.execute(query)
                if parameters is None
                else con.execute(query, parameters)
            )
            rows = cursor.fetchall()
        assert len(rows) == 1, rows
        assert len(rows[0]) == 1, rows[0]
        return rows[0][0]

    def _query_column(
        self, query: str, parameters: Sequence[Any] | None = None
    ) -> list[Any]:
        with sqlite3.connect(self._db_path) as con:
            cursor = (
                con.execute(query)
                if parameters is None
                else con.execute(query, parameters)
            )
            rows = cursor.fetchall()
        assert len(rows[0]) == 1, rows[0]
        return [row[0] for row in rows]

    def _query_rows(
        self, query: str, parameters: Sequence[Any] | None = None
    ) -> dict[str, list[Any]]:
        with sqlite3.connect(self._db_path) as con:
            cursor = (
                con.execute(query)
                if parameters is None
                else con.execute(query, parameters)
            )
            rows = cursor.fetchall()
            column_names = [t[0] for t in cursor.description]
        return self._query_result_to_dict(column_names, rows)

    def _query_rows_many(
        self, query: str, parameters: Sequence[Any]
    ) -> dict[str, list[Any]]:
        with sqlite3.connect(self._db_path) as con:
            cursor = con.executemany(query, parameters)
            rows = cursor.fetchall()
            column_names = [t[0] for t in cursor.description]
        return self._query_result_to_dict(column_names, rows)
