import hashlib
import io
import os
import re
import shutil
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage

from mldb.registry import Registry, TypeHandlerRegistry


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
