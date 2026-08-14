import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Environment:
    MLDB_DATA_ROOT: Path = Path(tempfile.mkdtemp(prefix="mldb_test_"))

    @classmethod
    def variables(cls) -> Iterator[tuple[str, Path]]:
        return ((f.name, getattr(cls, f.name)) for f in fields(cls))

    @classmethod
    def export_variables(cls) -> None:
        for k, v in cls.variables():
            os.environ[k] = str(v)
