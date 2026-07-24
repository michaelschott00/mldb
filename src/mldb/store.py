from __future__ import annotations

import os
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from os import environ
from pathlib import Path
from typing import Any, Self
from zoneinfo import ZoneInfo

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
from sqlalchemy.schema import CreateTable

from mldb.utils import generate_name

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


@dataclass
class StoredArtifact:
    run_id: str
    artifact_name: str
    artifact: Any


def _require_env(name: str) -> str:
    """Return the value of an environment variable, raising if it is unset."""
    val = environ.get(name)
    if val is None:
        raise EnvironmentError(f"Required environment variable '{name}' is not set.")
    return val


class BlobStore:
    def __init__(
        self,
        root_dir: str,
        hash_depth: int,
        hash_granularity: int,
    ) -> None:
        """Initialize the blob store, creating the root directory if needed."""
        self._hash_depth = hash_depth
        self._hash_granularity = hash_granularity
        self._root_dir = Path(root_dir)
        self._root_dir.mkdir(exist_ok=True, parents=True)

    def store(self, blob: Any) -> str:
        """Save a blob to disk using its type-specific handler and return its checksum."""
        from mldb import blobs

        handler = blobs.store_handlers.get_class(type(blob))
        checksum = handler.create_hash(blob)
        handler.save_to_disk(blob, self._uri(checksum))
        return checksum

    def load(self, checksum: str) -> Any:
        """Load and deserialize the blob identified by the given checksum."""
        from mldb import blobs

        uri = self._uri(checksum, resolve_ext=True)
        _, ext = os.path.splitext(os.path.basename(uri))
        return blobs.load_handlers.get_class(ext).load_from_disk(uri)

    def uri(self, checksum: str, resolve_ext: bool = False) -> str:
        """Return the on-disk path for a checksum, optionally resolving its file extension."""
        return self._uri(checksum, resolve_ext)

    def delete(self, checksum: str) -> None:
        """Delete the blob file identified by the given checksum."""
        os.remove(self._uri(checksum, resolve_ext=True))

    def _uri(self, checksum: str, resolve_ext: bool = False) -> str:
        """Compute the bucketed on-disk path for a checksum, optionally matching any file extension."""
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
        """Initialize the blob store and create the SQLite index tables if they don't exist."""
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
        """Dispose of the underlying database engine and its connections."""
        self._engine.dispose()

    def __del__(self) -> None:
        """Ensure the database engine is disposed when the store is garbage collected."""
        self.close()

    @classmethod
    def from_env(cls) -> Self:
        """Construct a RunStore using the root directory from the DATA_ROOT environment variable."""
        root_dir = _require_env("DATA_ROOT")
        return cls(root_dir=root_dir)

    def list_runs(
        self, include_tags: list[str], exclude_tags: list[str]
    ) -> list[RunInfo]:
        """List runs matching the given tag filters, including each run's tags."""
        tag_filter = self._get_tag_select(include_tags, exclude_tags).subquery()

        # Find run_ids matching the tag query
        stmt = select(self._runs).join(
            tag_filter, self._runs.c.run_id == tag_filter.c.run_id
        )
        with self._engine.connect() as conn:
            run_rows = conn.execute(stmt).all()
        if not run_rows:
            return []
        run_ids = [r.run_id for r in run_rows]

        # Find tags assigned to the matched runs to display them to the user and return result
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
        """List artifacts, optionally filtered to a single run."""
        stmt = select(self._artifacts)
        if run_id is not None:
            stmt = stmt.where(self._artifacts.c.run_id == run_id)
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).all()
        return [
            ArtifactInfo(r.run_id, r.artifact_name, r.artifact_checksum) for r in rows
        ]

    def create_run(self, tags: list[str] | None = None) -> str:
        """Create a new run with a generated ID and name, recording its timestamp and optional tags."""
        name = generate_name()
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
        """Attach the given tags to a run."""
        with self._engine.connect() as conn:
            conn.execute(
                insert(self._tags),
                [{"run_id": run_id, "tag": t} for t in tags],
            )
            conn.commit()

    def remove_tags(self, run_id: str, tags: list[str]) -> None:
        """Detach the given tags from a run."""
        with self._engine.connect() as conn:
            conn.execute(
                delete(self._tags).where(
                    self._tags.c.run_id == run_id,
                    self._tags.c.tag.in_(tags),
                )
            )
            conn.commit()

    def delete_run(self, run_id: str) -> None:
        """Delete a run, its tags, and its artifact records, removing any blobs no longer referenced by other runs."""
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
        """Store each artifact's blob and record its checksum under the given run."""
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
        """Load a single artifact's blob by run ID and artifact name."""
        stmt = select(self._artifacts.c.artifact_checksum).where(
            self._artifacts.c.run_id == run_id,
            self._artifacts.c.artifact_name == artifact_name,
        )
        with self._engine.connect() as conn:
            checksum = conn.execute(stmt).scalar_one_or_none()
        if checksum is None:
            raise ValueError(f"Artifact {artifact_name} not found for {run_id}")
        return self._blob_store.load(checksum)

    def load_by_tags_single(
        self,
        artifact_name: str,
        include_tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
    ) -> StoredArtifact:
        """Load the single blob of an artifact matching the given tag filters, raising if there isn't exactly one match."""
        results = self.load_by_tags_all(artifact_name, include_tags, exclude_tags)
        if len(results) == 0:
            raise ValueError(
                f"Artifact {artifact_name} not found for tag query {include_tags}, {exclude_tags}"
            )
        if len(results) > 1:
            raise ValueError(
                f"Tag query {include_tags}, {exclude_tags} returned more than one result for {artifact_name}."
                "Change tag query or use load_by_tags_all to get a list of all matches."
            )
        return results[0]

    def load_by_tags_all(
        self,
        artifact_name: str,
        include_tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
    ) -> list[StoredArtifact]:
        """Load blobs of an artifact across all runs matching the given tag filters."""
        assert not (include_tags is None and exclude_tags is None), (
            "Must provide either include_tags, exclude_tags or both."
        )
        if include_tags is None:
            include_tags = list()
        if exclude_tags is None:
            exclude_tags = list()
        results = list()
        for row in self._search_artifacts_by_tags(
            artifact_name, include_tags, exclude_tags
        ):
            assert row.run_id not in results, results
            results.append(
                StoredArtifact(
                    row.run_id,
                    artifact_name,
                    self._blob_store.load(row.artifact_checksum),
                )
            )
        return results

    @contextmanager
    def load_duckdb(
        self, *args: tuple[str, tuple[str, ...], tuple[str, ...]]
    ) -> Generator[Any]:
        """Yield a DuckDB connection with tables created from artifacts matching the given (name, include_tags, exclude_tags) specs."""
        import duckdb

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
        """Build a select statement for run IDs having all include_tags and none of exclude_tags."""
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
        """Fetch artifact rows for the given artifact name(s) across runs matching the tag filters."""
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
