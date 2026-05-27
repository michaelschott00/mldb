import hashlib
import io
import os
import re
import shutil
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
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
from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    insert,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoResultFound
from sqlalchemy.schema import CreateTable

from mldb.registry import Registry, TypeHandlerRegistry

DEFAULT_HASH_DEPTH = 3
DEFAULT_HASH_GRANULARITY = 1


@dataclass
class RunInfo:
    run_id: str
    run_name: str
    run_timestamp: str
    tags: list[str] = field(default_factory=list)


@dataclass
class ArtifactInfo:
    run_id: str
    artifact_name: str
    artifact_checksum: str


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


class BlobStore:
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

    def store(self, blob: Any) -> str:
        checksum = self._checksum(blob)
        store_handlers.get_class(type(blob)).save_to_disk(blob, self._uri(checksum))
        return checksum

    def load(self, checksum: str) -> Any:
        uri = self._uri(checksum, resolve_ext=True)
        _, ext = os.path.splitext(os.path.basename(uri))
        return load_handlers.get_class(ext).load_from_disk(uri)

    def uri(self, checksum: str, resolve_ext: bool = False) -> str:
        return self._uri(checksum, resolve_ext)

    def delete(self, checksum: str) -> None:
        os.remove(self._uri(checksum, resolve_ext=True))

    def _checksum(self, blob: Any) -> str:
        return store_handlers.get_class(type(blob)).create_hash(blob)

    def _uri(self, checksum: str, resolve_ext: bool = False) -> str:
        n_digits = self._hash_granularity * 2
        bucket = [
            checksum[i * n_digits : (i + 1) * n_digits] for i in range(self._hash_depth)
        ]
        bucket_dir = self._root_dir / Path(*bucket)
        bucket_dir.mkdir(exist_ok=True, parents=True)
        if resolve_ext:
            matches = list(bucket_dir.glob(checksum + "*"))
            if len(matches) == 0:
                raise ValueError("No match.")
            if len(matches) > 1:
                raise ValueError("Multiple matches.")
            return str(matches[0])
        return str(bucket_dir / checksum)


class RunStore:
    def __init__(
        self,
        root_dir: str,
        hash_depth: int = DEFAULT_HASH_DEPTH,
        hash_granularity: int = DEFAULT_HASH_GRANULARITY,
    ) -> None:
        self._blob_store = BlobStore(root_dir, hash_depth, hash_granularity)

        db_path = Path(root_dir) / "index.db"
        self._engine: Engine = create_engine(f"sqlite:///{db_path}")
        _meta = MetaData()
        self._artifacts = Table(
            "artifacts",
            _meta,
            Column("run_id", String, primary_key=True),
            Column("artifact_name", String, primary_key=True),
            Column("artifact_checksum", String, nullable=False),
        )
        self._runs = Table(
            "runs",
            _meta,
            Column("run_id", String, primary_key=True),
            Column("run_name", String, nullable=False),
            Column("run_timestamp", String, nullable=False),
        )
        self._tags = Table(
            "tags",
            _meta,
            Column("run_id", String, primary_key=True),
            Column("tag", String, primary_key=True),
        )
        with self._engine.connect() as conn:
            for table in [self._artifacts, self._runs, self._tags]:
                conn.execute(CreateTable(table, if_not_exists=True))
            conn.commit()

    def close(self) -> None:
        self._engine.dispose()

    def __del__(self) -> None:
        self.close()

    @classmethod
    def from_env(cls) -> Self:
        root_dir = _require_env("DATA_ROOT")
        return cls(root_dir=root_dir)

    def list_runs(
        self, include_tags: list[str], exclude_tags: list[str]
    ) -> list[RunInfo]:
        tag_filter = self._get_tag_select(include_tags, exclude_tags).subquery()
        stmt = select(self._runs).join(
            tag_filter, self._runs.c.run_id == tag_filter.c.run_id
        )
        with self._engine.connect() as conn:
            run_rows = conn.execute(stmt).all()
        if not run_rows:
            return []
        run_ids = [r.run_id for r in run_rows]
        tags_stmt = select(self._tags).where(self._tags.c.run_id.in_(run_ids))
        with self._engine.connect() as conn:
            tag_rows = conn.execute(tags_stmt).all()
        tags_by_run: dict[str, list[str]] = {r.run_id: [] for r in run_rows}
        for row in tag_rows:
            tags_by_run[row.run_id].append(row.tag)
        return [
            RunInfo(r.run_id, r.run_name, r.run_timestamp, tags_by_run[r.run_id])
            for r in run_rows
        ]

    def list_artifacts(self, run_id: str | None = None) -> list[ArtifactInfo]:
        stmt = select(self._artifacts)
        if run_id is not None:
            stmt = stmt.where(self._artifacts.c.run_id == run_id)
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).all()
        return [
            ArtifactInfo(r.run_id, r.artifact_name, r.artifact_checksum) for r in rows
        ]

    def create_run(self, name: str, tags: list[str] | None = None) -> str:
        timezone = ZoneInfo("Europe/Berlin")
        timestamp = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
        run_id = f"run_{timestamp.replace(' ', '-')}_{uuid.uuid4().hex[:8]}_{name}"
        with self._engine.connect() as conn:
            conn.execute(
                insert(self._runs).values(
                    run_id=run_id, run_name=name, run_timestamp=timestamp
                )
            )
            conn.commit()
        if tags is not None:
            self.add_tags(run_id, tags)
        return run_id

    def add_tags(self, run_id: str, tags: list[str]) -> None:
        with self._engine.connect() as conn:
            conn.execute(
                insert(self._tags),
                [{"run_id": run_id, "tag": t} for t in tags],
            )
            conn.commit()

    def remove_tags(self, run_id: str, tags: list[str]) -> None:
        with self._engine.connect() as conn:
            conn.execute(
                delete(self._tags).where(
                    self._tags.c.run_id == run_id,
                    self._tags.c.tag.in_(tags),
                )
            )
            conn.commit()

    def delete_run(self, run_id: str) -> None:
        orphaned_stmt = select(self._artifacts.c.artifact_checksum).where(
            self._artifacts.c.run_id == run_id,
            ~self._artifacts.c.artifact_checksum.in_(
                select(self._artifacts.c.artifact_checksum).where(
                    self._artifacts.c.run_id != run_id
                )
            ),
        )
        with self._engine.connect() as conn:
            orphaned_checksums = list(conn.execute(orphaned_stmt).scalars())
        with self._engine.connect() as conn:
            conn.execute(delete(self._runs).where(self._runs.c.run_id == run_id))
            conn.execute(delete(self._tags).where(self._tags.c.run_id == run_id))
            conn.execute(
                delete(self._artifacts).where(self._artifacts.c.run_id == run_id)
            )
            conn.commit()
        for checksum in orphaned_checksums:
            self._blob_store.delete(checksum)

    def store(self, run_id: str, artifacts: dict[int | str, Any]) -> None:
        artifact_rows = []
        for key, blob in artifacts.items():
            checksum = self._blob_store.store(blob)
            artifact_rows.append(
                {
                    "run_id": run_id,
                    "artifact_name": str(key),
                    "artifact_checksum": checksum,
                }
            )
        with self._engine.connect() as conn:
            conn.execute(insert(self._artifacts), artifact_rows)
            conn.commit()

    def load(self, run_id: str, artifact_name: str) -> Any:
        stmt = select(self._artifacts.c.artifact_checksum).where(
            self._artifacts.c.run_id == run_id,
            self._artifacts.c.artifact_name == artifact_name,
        )
        with self._engine.connect() as conn:
            checksum = conn.execute(stmt).scalar_one_or_none()
        if checksum is None:
            raise ValueError(f"Artifact {artifact_name} not found for {run_id}")
        return self._blob_store.load(checksum)

    def load_by_tags(
        self, artifact_name: str, include_tags: list[str], exclude_tags: list[str]
    ) -> dict[str, Any]:
        result = dict()
        for row in self._search_artifacts_by_tags(
            artifact_name, include_tags, exclude_tags
        ):
            assert row.run_id not in result, result
            result[row.run_id] = self._blob_store.load(row.artifact_checksum)
        return result

    @contextmanager
    def load_duckdb(
        self, *args: tuple[str, tuple[str, ...], tuple[str, ...]]
    ) -> Generator[duckdb.DuckDBPyConnection]:
        con = duckdb.connect()
        try:
            for artifact_name, include_tags, exclude_tags in args:
                for row in self._search_artifacts_by_tags(
                    artifact_name, list(include_tags), list(exclude_tags)
                ):
                    uri = self._blob_store.uri(row.artifact_checksum, resolve_ext=True)
                    assert uri.endswith(".csv")
                    # TODO: Dangerous string replacement
                    con.sql(f"create table {row.artifact_name} as from '{uri}'")
            yield con
        finally:
            con.close()

    def _get_tag_select(self, include_tags: list[str], exclude_tags: list[str]):
        stmt = select(self._runs.c.run_id)
        for tag in include_tags:
            stmt = stmt.where(
                self._runs.c.run_id.in_(
                    select(self._tags.c.run_id).where(self._tags.c.tag == tag)
                )
            )
        for tag in exclude_tags:
            stmt = stmt.where(
                ~self._runs.c.run_id.in_(
                    select(self._tags.c.run_id).where(self._tags.c.tag == tag)
                )
            )
        return stmt

    def _search_artifacts_by_tags(
        self,
        artifact_name: str | list[str],
        include_tags: list[str],
        exclude_tags: list[str],
    ):
        artifact_names = (
            [artifact_name] if isinstance(artifact_name, str) else artifact_name
        )
        tag_cte = self._get_tag_select(include_tags, exclude_tags).cte()
        stmt = (
            select(
                self._artifacts.c.run_id,
                self._artifacts.c.artifact_name,
                self._artifacts.c.artifact_checksum,
            )
            .join(tag_cte, self._artifacts.c.run_id == tag_cte.c.run_id)
            .where(self._artifacts.c.artifact_name.in_(artifact_names))
        )
        with self._engine.connect() as conn:
            return conn.execute(stmt).all()
