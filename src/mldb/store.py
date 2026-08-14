from __future__ import annotations

import os
import shutil
import uuid
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from os import environ
from pathlib import Path
from typing import Any, Self
from zoneinfo import ZoneInfo

from sqlalchemy import (
    CTE,
    Column,
    ForeignKey,
    MetaData,
    String,
    Table,
    and_,
    create_engine,
    delete,
    insert,
    or_,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import aliased
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
    hparams: list[str] = field(default_factory=list)


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
        p = Path(self._uri(checksum, resolve_ext=True))
        if p.is_dir():
            import shutil

            shutil.rmtree(p)
        else:
            p.unlink()

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
        self._hash_depth = hash_depth
        self._hash_granularity = hash_granularity
        self._blob_store = BlobStore(root_dir, hash_depth, hash_granularity)
        db_path = Path(root_dir) / "index.db"
        self._engine: Engine = create_engine(f"sqlite:///{db_path}")
        _meta = MetaData()
        self._store_configuration = Table(
            "store_configuration",
            _meta,
            Column("hash_depth", String, primary_key=True),
            Column("hash_granularity", String, primary_key=True),
        )
        self._artifacts = Table(
            "artifacts",
            _meta,
            Column("run_id", String, ForeignKey("runs.run_id"), primary_key=True),
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
            Column("run_id", String, ForeignKey("runs.run_id"), primary_key=True),
            Column("tag", String, primary_key=True),
        )
        self._hparams = Table(
            "hparams",
            _meta,
            Column("run_id", String, ForeignKey("runs.run_id"), primary_key=True),
            Column("name", String, primary_key=True),
            Column("type", String, primary_key=False, nullable=False),
            Column("value", String, primary_key=False, nullable=False),
        )
        with self._engine.connect() as conn:
            for table in [
                self._store_configuration,
                self._artifacts,
                self._runs,
                self._tags,
                self._hparams,
            ]:
                conn.execute(CreateTable(table, if_not_exists=True))
            conn.commit()
        self._check_or_write_configuration(root_dir)

    def _check_or_write_configuration(self, root_dir: str) -> None:
        """Ensure the store's configuration matches the settings recorded on disk, writing them on first use."""
        with self._engine.connect() as conn:
            row = conn.execute(select(self._store_configuration)).first()
            if row is None:
                conn.execute(
                    insert(self._store_configuration).values(
                        hash_depth=str(self._hash_depth),
                        hash_granularity=str(self._hash_granularity),
                    )
                )
                conn.commit()
                return
        stored_depth, stored_granularity = (
            int(row.hash_depth),
            int(row.hash_granularity),
        )
        if (stored_depth, stored_granularity) != (
            self._hash_depth,
            self._hash_granularity,
        ):
            raise ValueError(
                f"Store at '{root_dir}' was created with hash_depth={stored_depth}, "
                f"hash_granularity={stored_granularity}, but was opened with "
                f"hash_depth={self._hash_depth}, hash_granularity={self._hash_granularity}."
            )

    def __del__(self) -> None:
        """Ensure the database engine is disposed when the store is garbage collected."""
        self.close()

    def close(self) -> None:
        """Dispose of the underlying database engine and its connections."""
        self._engine.dispose()

    @classmethod
    def from_env(cls) -> Self:
        """Construct a RunStore using the root directory from the MLDB_DATA_ROOT environment variable."""
        root_dir = _require_env("MLDB_DATA_ROOT")
        return cls(root_dir=root_dir)

    def create_run(
        self, hparams: dict[str, Any] | None = None, tags: list[str] | None = None
    ) -> str:
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
        if hparams is not None:
            self.add_hparams(run_id, hparams)
        return run_id

    def list_runs(
        self,
        hparams: dict[str, list[Any]] | None = None,
        tags: list[str] | None = None,
    ) -> list[RunInfo]:
        """List runs matching the given tag filters, including each run's tags."""
        # Find run_ids matching the tag query
        stmt = select(self._runs)
        if tags is not None and len(tags) > 0:
            stmt = (
                stmt.distinct()
                .join(self._tags, self._runs.c.run_id == self._tags.c.run_id)
                .where(self._tags.c.tag.in_(tags))
            )
        if hparams is not None and len(hparams) > 0:
            hparams_cte = self._get_hparams_cte(hparams)
            conds = [
                getattr(hparams_cte.c, name).in_([str(v) for v in values])
                for name, values in hparams.items()
            ]
            stmt = stmt.join(
                hparams_cte,
                self._runs.c.run_id == hparams_cte.c.run_id,
            ).where(and_(*conds))
        with self._engine.connect() as conn:
            run_rows = conn.execute(stmt).all()
        if not run_rows:
            return []
        run_ids = [r.run_id for r in run_rows]

        # Find tags and hparams assigned to the matched runs to display them to the user and return result
        tags_stmt = select(self._tags).where(self._tags.c.run_id.in_(run_ids))
        hparams_stmt = select(self._hparams).where(self._hparams.c.run_id.in_(run_ids))
        with self._engine.connect() as conn:
            tag_rows = conn.execute(tags_stmt).all()
            hparam_rows = conn.execute(hparams_stmt).all()
        tags_by_run: dict[str, list[str]] = {r.run_id: [] for r in run_rows}
        hparams_by_run: dict[str, list[str]] = {r.run_id: [] for r in run_rows}
        for row in tag_rows:
            tags_by_run[row.run_id].append(row.tag)
        for row in hparam_rows:
            hparams_by_run[row.run_id].append(f"{row.name}={row.value}")

        return [
            RunInfo(
                r.run_id,
                r.run_name,
                r.run_timestamp,
                tags_by_run[r.run_id],
                hparams_by_run[r.run_id],
            )
            for r in run_rows
        ]

    def add_tags(self, run_id: str, tags: list[str]) -> None:
        """Attach the given tags to a run."""
        with self._engine.connect() as conn:
            conn.execute(
                insert(self._tags),
                [{"run_id": run_id, "tag": t} for t in tags],
            )
            conn.commit()

    def add_hparams(self, run_id: str, hparams: dict[str, Any]) -> None:
        """Attach the given hyperparameter settings to a run."""
        with self._engine.connect() as conn:
            conn.execute(
                insert(self._hparams),
                [
                    {
                        "run_id": run_id,
                        "name": k,
                        "type": v.__class__.__name__,
                        "value": str(v),
                    }
                    for k, v in hparams.items()
                ],
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

    def remove_hparams(self, run_id: str, hparams: list[str]) -> None:
        """Detach the given hyperparameters from a run."""
        with self._engine.connect() as conn:
            conn.execute(
                delete(self._hparams).where(
                    self._tags.c.run_id == run_id,
                    self._tags.c.name.in_(hparams),
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

    def merge(self, other_root_dir: str) -> None:
        """Merge all runs, tags, hparams, artifacts and blobs from another store's directory into this one.

        Raises if the other store was created with different hash_depth/hash_granularity
        settings than this one (checked via each store's store_configuration table).
        """
        try:
            other = RunStore(
                root_dir=other_root_dir,
                hash_depth=self._hash_depth,
                hash_granularity=self._hash_granularity,
            )
        except ValueError as e:
            raise ValueError(f"Cannot merge store at '{other_root_dir}': {e}") from e
        try:
            with other._engine.connect() as conn:
                run_rows = [dict(r._mapping) for r in conn.execute(select(other._runs))]
                tag_rows = [dict(r._mapping) for r in conn.execute(select(other._tags))]
                hparam_rows = [
                    dict(r._mapping) for r in conn.execute(select(other._hparams))
                ]
                artifact_rows = [
                    dict(r._mapping) for r in conn.execute(select(other._artifacts))
                ]

            checksums = {r["artifact_checksum"] for r in artifact_rows}
            for checksum in checksums:
                self._copy_blob(other._blob_store, checksum)

            with self._engine.connect() as conn:
                if run_rows:
                    conn.execute(insert(self._runs).prefix_with("OR IGNORE"), run_rows)
                if tag_rows:
                    conn.execute(insert(self._tags).prefix_with("OR IGNORE"), tag_rows)
                if hparam_rows:
                    conn.execute(
                        insert(self._hparams).prefix_with("OR IGNORE"), hparam_rows
                    )
                if artifact_rows:
                    conn.execute(
                        insert(self._artifacts).prefix_with("OR IGNORE"), artifact_rows
                    )
                conn.commit()
        finally:
            other.close()

    def _copy_blob(self, other_blob_store: BlobStore, checksum: str) -> None:
        """Copy a single blob (file or directory) from another blob store into this one, skipping if already present."""
        try:
            src = Path(other_blob_store.uri(checksum, resolve_ext=True))
        except ValueError:
            return
        dst = Path(self._blob_store.uri(checksum)).parent / src.name
        if dst.exists():
            return
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copyfile(src, dst)

    def export(
        self,
        dest_dir: str,
        tags: list[str] | None = None,
        hparams: dict[str, list[Any]] | None = None,
    ) -> None:
        """Export the runs matching the given tag/hparam filters into a new store directory.

        The destination directory contains only the matched runs, their tags, hparams,
        artifacts and blobs, and can be imported into another store via merge().
        """
        matched_runs = self.list_runs(tags=tags, hparams=hparams)
        run_ids = {r.run_id for r in matched_runs}
        if not run_ids:
            raise ValueError("No runs matched the given tags/hparams for export.")

        with self._engine.connect() as conn:
            run_rows = [
                dict(r._mapping)
                for r in conn.execute(select(self._runs)).all()
                if r.run_id in run_ids
            ]
            tag_rows = [
                dict(r._mapping)
                for r in conn.execute(select(self._tags)).all()
                if r.run_id in run_ids
            ]
            hparam_rows = [
                dict(r._mapping)
                for r in conn.execute(select(self._hparams)).all()
                if r.run_id in run_ids
            ]
            artifact_rows = [
                dict(r._mapping)
                for r in conn.execute(select(self._artifacts)).all()
                if r.run_id in run_ids
            ]

        dest = RunStore(
            root_dir=dest_dir,
            hash_depth=self._hash_depth,
            hash_granularity=self._hash_granularity,
        )
        try:
            checksums = {r["artifact_checksum"] for r in artifact_rows}
            for checksum in checksums:
                dest._copy_blob(self._blob_store, checksum)

            with dest._engine.connect() as conn:
                if run_rows:
                    conn.execute(insert(dest._runs), run_rows)
                if tag_rows:
                    conn.execute(insert(dest._tags), tag_rows)
                if hparam_rows:
                    conn.execute(insert(dest._hparams), hparam_rows)
                if artifact_rows:
                    conn.execute(insert(dest._artifacts), artifact_rows)
                conn.commit()
        finally:
            dest.close()

    def store(self, run_id: str, artifacts: dict[int | str, Any]) -> None:
        """Store each artifact's blob and record its checksum under the given run.

        Supported artifact types:
            - np.ndarray
            - str (path of a file with arbitrary filetype)
            - pd.DataFrame
            - PIL.Image.Image
            - torch.Tensor
            - dict[str, Any] (torch state dict)"""
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

    def open_directory(self, run_id: str) -> str:
        checksum = run_id[::-1]
        run_dir_uri = self._blob_store.uri(checksum)  # prefix is always run_20...
        run_dir_path = Path(run_dir_uri)

        # Check if directory is in the database and in the filesystem
        with self._engine.connect() as conn:
            stored_checksums = conn.execute(
                select(self._artifacts).where(
                    self._artifacts.c.artifact_checksum == checksum
                )
            ).all()
        if run_dir_path.exists() and len(stored_checksums) == 1:
            return run_dir_uri
        if run_dir_path.exists() or len(stored_checksums) != 0:
            raise ValueError(
                f"Blob store has been corrupted: {stored_checksums}, {run_dir_path}."
            )

        # Create directory if it doesn't exist
        run_dir_path.mkdir()
        with self._engine.connect() as conn:
            conn.execute(
                insert(self._artifacts),
                {
                    "run_id": run_id,
                    "artifact_name": "directory",
                    "artifact_checksum": checksum,
                },
            )
            conn.commit()
        return run_dir_uri

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

    def list_artifacts_by_run(self, run_id: str | None = None) -> list[ArtifactInfo]:
        """List artifacts, optionally filtered to a single run."""
        stmt = select(self._artifacts)
        if run_id is not None:
            stmt = stmt.where(self._artifacts.c.run_id == run_id)
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).all()
        return [
            ArtifactInfo(r.run_id, r.artifact_name, r.artifact_checksum) for r in rows
        ]

    def list_directory_by_run(self, run_id: str) -> str:
        stmt = select(self._artifacts).where(
            self._artifacts.c.run_id == run_id,
            self._artifacts.c.artifact_name == "directory",
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).all()
        if len(rows) == 0:
            raise ValueError(f"No directory found for run {run_id}.")
        assert len(rows) == 1, rows
        return self._blob_store.uri(rows[0].artifact_checksum)

    def list_artifacts_by_query(
        self,
        names: list[str] | None = None,
        hparams: dict[str, list[Any]] | None = None,
        tags: list[str] | None = None,
    ) -> list[ArtifactInfo]:
        """Fetch artifact rows across runs matching the tag filters, optionally restricted to the given artifact name(s). If artifact_name is None, all artifact names are matched."""
        if names is None and hparams is None and tags is None:
            raise ValueError("Must specify at least one of names, hparams or tags")
        stmt = select(self._artifacts).distinct()
        if names is not None and len(names) > 0:
            stmt = stmt.where(self._artifacts.c.artifact_name.in_(names))
        if tags is not None and len(tags) > 0:
            stmt = stmt.join(
                self._tags, self._tags.c.run_id == self._artifacts.c.run_id
            ).where(self._tags.c.tag.in_(tags))
        if hparams is not None and len(hparams) > 0:
            hparams_cte = self._get_hparams_cte(hparams)
            conds = [
                getattr(hparams_cte.c, name).in_([str(v) for v in values])
                for name, values in hparams.items()
            ]
            stmt = stmt.join(
                hparams_cte,
                self._artifacts.c.run_id == hparams_cte.c.run_id,
            ).where(and_(*conds))
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).all()
        return [
            ArtifactInfo(r.run_id, r.artifact_name, r.artifact_checksum) for r in rows
        ]

    def load_artifacts_by_query(
        self,
        names: list[str] | None = None,
        hparams: dict[str, list[Any]] | None = None,
        tags: list[str] | None = None,
    ) -> list[StoredArtifact]:
        """Load blobs of an artifact across all runs matching the given tag filters."""
        if isinstance(names, str):
            raise ValueError(
                "names argument takes a list of strings, even if only one name is provided."
            )
        results = list()
        for row in self.list_artifacts_by_query(names, hparams, tags):
            assert row.run_id not in results, results
            results.append(
                StoredArtifact(
                    row.run_id,
                    row.artifact_name,
                    self._blob_store.load(row.artifact_checksum),
                )
            )
        return results

    def load_artifact_by_query(
        self,
        name: str,
        hparams: dict[str, list[Any]] | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        """Load the single blob of an artifact matching the given tag filters, raising if there isn't exactly one match."""
        results = self.load_artifacts_by_query([name], hparams, tags)
        if len(results) == 0:
            raise ValueError(
                f"Artifacts {name} not found for tag query {hparams}, {tags}"
            )
        if len(results) > 1:
            raise ValueError(
                f"Tag query {hparams}, {tags} returned more than one result for {name}."
                "Change tag query or use load_artifacts_by_tags to get a list of all matches."
            )
        return results[0].artifact

    def list_directories_by_query(
        self,
        hparams: dict[str, list[Any]] | None = None,
        tags: list[str] | None = None,
    ) -> list[str]:
        results = self.list_artifacts_by_query(["directory"], hparams, tags)
        return [self._blob_store.uri(a.artifact_checksum) for a in results]

    def get_db(self) -> ResultsDatabase:
        return ResultsDatabase(self)

    def get_tags(self, run_ids: list[str]) -> list[dict[str, str]]:
        """Return raw (run_id, tag) rows restricted to the given run_ids."""
        stmt = select(self._tags).where(self._tags.c.run_id.in_(run_ids))
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).all()
        return [dict(r._mapping) for r in rows]

    def get_hparams(self, run_ids: list[str]) -> list[dict[str, str]]:
        """Return raw (run_id, name, type, value) hparam rows restricted to the given run_ids."""
        stmt = select(self._hparams).where(self._hparams.c.run_id.in_(run_ids))
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).all()
        return [dict(r._mapping) for r in rows]

    def _get_hparams_cte(self, hparams: dict[str, list[str]]) -> CTE:
        """Flatten hparam EAV table using JOINs and return resulting table as CTE"""
        hparam_names = list(hparams.keys())
        hparam_tables = [aliased(self._hparams) for _ in range(len(hparam_names))]
        cte = select(
            hparam_tables[0].c.run_id,
            *[t.c.value.label(n) for t, n in zip(hparam_tables, hparam_names)],
        )
        if len(hparams) == 1:
            cte = cte.where(hparam_tables[0].c.name == hparam_names[0])
        else:
            for table, name in zip(hparam_tables[1:], hparam_names[1:]):
                cte = cte.join_from(
                    hparam_tables[0],
                    table,
                    and_(
                        hparam_tables[0].c.run_id == table.c.run_id,
                        hparam_tables[0].c.name == hparam_names[0],
                        table.c.name == name,
                    ),
                )
        cte = cte.cte()
        return cte


class ResultsDatabase:
    def __init__(self, store: RunStore) -> None:
        self._store = store
        self._tables: list[ArtifactInfo] = list()

    def attach(
        self,
        names: list[str] | None,
        hparams: dict[str, list[Any]] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        for artifact in self._store.list_artifacts_by_query(names, hparams, tags):
            if artifact not in self._tables:
                self._tables.append(artifact)

    @contextmanager
    def connect(self) -> Generator[Any]:
        """Yield a DuckDB connection with tables created from all CSV artifacts matching the given tag filters, augmented with a column per tag and hparam."""
        import duckdb

        num_attached = 0
        con = duckdb.connect()
        try:
            artifact_names: set[str] = set()
            for row in self._tables:
                uri = self._store._blob_store.uri(
                    row.artifact_checksum, resolve_ext=True
                )
                if not uri.endswith(".csv"):
                    continue
                num_attached += 1
                # TODO: Dangerous string replacement
                try:
                    con.sql(
                        f"CREATE TABLE {row.artifact_name} AS "
                        f"SELECT *, '{row.run_id}' AS run_id FROM '{uri}'"
                    )
                except duckdb.CatalogException:
                    try:
                        con.sql(
                            f"INSERT INTO {row.artifact_name} BY NAME "
                            f"SELECT *, '{row.run_id}' AS run_id FROM '{uri}'"
                        )
                    except duckdb.BinderException:
                        raise ValueError(
                            f"Matches for artifact {row.artifact_name} have differing schemas."
                        )
                artifact_names.add(row.artifact_name)
            if num_attached == 0:
                warnings.warn("No tables attached. Database will be empty.")

            run_ids = sorted({row.run_id for row in self._tables})
            if run_ids and artifact_names:
                self._attach_metadata(con, run_ids, artifact_names)

            yield con
        finally:
            con.close()

    def _attach_metadata(
        self, con: Any, run_ids: list[str], artifact_names: set[str]
    ) -> None:
        """Pivot tags/hparams (filtered to run_ids) into one row per run_id and left-join it onto every artifact table."""
        tag_rows = self._store.get_tags(run_ids)
        hparam_rows = self._store.get_hparams(run_ids)

        metadata_tables: list[str] = []

        if tag_rows:
            con.sql("CREATE TEMP TABLE _tags_raw (run_id VARCHAR, tag VARCHAR)")
            con.executemany(
                "INSERT INTO _tags_raw VALUES (?, ?)",
                [(r["run_id"], r["tag"]) for r in tag_rows],
            )
            # One row per run_id, one column per tag; value is the tag's row count
            # for that run (nonzero == tag present, since tags are unique per run).
            con.sql(
                "CREATE TEMP TABLE _tags_pivot AS "
                "PIVOT _tags_raw ON tag USING count(tag) GROUP BY run_id"
            )
            metadata_tables.append("_tags_pivot")

        if hparam_rows:
            con.sql(
                "CREATE TEMP TABLE _hparams_raw (run_id VARCHAR, name VARCHAR, value VARCHAR)"
            )
            con.executemany(
                "INSERT INTO _hparams_raw VALUES (?, ?, ?)",
                [(r["run_id"], r["name"], r["value"]) for r in hparam_rows],
            )
            con.sql(
                "CREATE TEMP TABLE _hparams_pivot AS "
                "PIVOT _hparams_raw ON name USING first(value) GROUP BY run_id"
            )
            metadata_tables.append("_hparams_pivot")

        if not metadata_tables:
            return

        if len(metadata_tables) == 2:
            con.sql(
                "CREATE TEMP TABLE _metadata AS "
                "SELECT * FROM _tags_pivot FULL OUTER JOIN _hparams_pivot USING (run_id)"
            )
        else:
            con.sql(
                f"CREATE TEMP TABLE _metadata AS SELECT * FROM {metadata_tables[0]}"
            )

        for name in artifact_names:
            con.sql(
                f"CREATE OR REPLACE TABLE {name} AS "
                f"SELECT * FROM {name} LEFT JOIN _metadata USING (run_id)"
            )
