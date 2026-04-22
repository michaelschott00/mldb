"""
This module defines all sorts of configurations.

Instances of each of the classes defined here can be stored in the database with the same schema.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import (
    MISSING,
    Field,
    dataclass,
    field,
    fields,
    is_dataclass,
)
from enum import EnumType, StrEnum, auto
from types import UnionType
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Self,
    Tuple,
    TypedDict,
    Union,
    get_args,
    get_origin,
)

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from discovery.db import MLDataset, ModelDB
from discovery.gates import UCBMActivationFnType


@dataclass
class Config:
    def _collect_df_fields(self) -> List[Tuple[str, str, Any]]:
        """Collect all fields as (class_name, col_name, value) triples, recursing into nested Configs."""
        ignored = {p.name for p in fields(self) if not p.metadata.get("df", True)}
        result = []
        for param in fields(self):
            if param.name in ignored:
                continue
            value = getattr(self, param.name)
            if isinstance(value, Config):
                result.extend(value._collect_df_fields())
            else:
                result.append((type(self).__name__, param.name, value))
        return result

    def to_df(self) -> pd.DataFrame:
        triples = self._collect_df_fields()

        # Count occurrences of each column name
        name_counts: Dict[str, int] = {}
        for _, col_name, _ in triples:
            name_counts[col_name] = name_counts.get(col_name, 0) + 1

        # Build final column names, prefixing duplicates with class name
        data = {}
        for cls_name, col_name, value in triples:
            if name_counts[col_name] > 1:
                data[f"{cls_name}.{col_name}"] = value
            else:
                data[col_name] = value

        return pd.DataFrame([data])

    @classmethod
    def from_args(cls, args: argparse.Namespace, **kwargs) -> Self:
        """Create a config object from command line arguments."""
        cls_params = {param.name for param in fields(cls)}
        cls_args = {
            k: v for k, v in vars(args).items() if k in cls_params and k not in kwargs
        }
        cls_args = {k.split(".")[-1]: v for k, v in cls_args.items()}
        return cls(**cls_args, **kwargs)  # type: ignore

    @classmethod
    def _collect_args(
        cls, params: Sequence[Field], result: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Collect all arguments that are not default values."""
        # Base cases
        if len(params) == 0:
            return result

        param = params[0]
        if param.metadata.get("argparse", True):
            field_type = param.type

            # Recursively add nested dataclass fields
            if (
                isinstance(field_type, type)
                and is_dataclass(field_type)
                and issubclass(field_type, Config)
            ):
                field_type._collect_args(params=fields(field_type), result=result)
            else:
                is_optional = False
                is_list = False
                choices = None
                is_dict = False
                origin_type = get_origin(field_type)
                # Handle Optional types
                if origin_type is Union or origin_type is UnionType:
                    type_args = get_args(field_type)
                    if any([t is type(None) for t in type_args]):
                        is_optional = True
                        new_args = [arg for arg in type_args if arg is not type(None)]
                        assert len(new_args) == 1, new_args
                        field_type = type_args[0]
                        origin_type = get_origin(field_type)
                    assert field_type is not type(None)

                # Handle other non-trivial types
                if origin_type is Literal:
                    choices = get_args(field_type)
                elif isinstance(field_type, EnumType):
                    choices = [str(option) for option in field_type]
                elif origin_type is list:
                    is_list = True
                    list_args = get_args(field_type)
                    if list_args:
                        field_type = list_args[0]
                    else:
                        field_type = str  # Default to str if no type arg
                elif origin_type is dict:
                    is_dict = True
                    field_type = [t.__name__ for t in get_args(field_type)]

                # Add a help message
                help_msg_items = list()
                if isinstance(field_type, type) and choices is None:
                    help_msg_items.append(f"Type: {field_type.__name__}")
                if param.default != MISSING:
                    is_optional = True
                    help_msg_items.append(f"Default: {param.default}")
                help_msg = ", ".join(help_msg_items)

                # Store parsed option
                option_name = param.name.replace("_", "-")
                if option_name not in result:
                    result[option_name] = []
                result[option_name].append(
                    {
                        "type": field_type,
                        "choices": choices,
                        "is_optional": is_optional,
                        "is_list": is_list,
                        "is_dict": is_dict,
                        "help_msg": help_msg,
                        "group": cls.__name__,
                        "default": param.default if param.default != MISSING else None,
                    }
                )

        return cls._collect_args(params=params[1:], result=result)

    @classmethod
    def add_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        """Add dataclass fields as arguments to an ArgumentParser.

        Args:
            parser: The ArgumentParser to add arguments to
            cls: The dataclass to extract fields from
        """
        class_options = cls._collect_args(params=fields(cls), result={})

        group_parsers = dict()
        for option_name, options in class_options.items():
            for option in options:
                # Prefix colliding options with the group name
                if len(options) > 1:
                    option_name = f"{option['group']}.{option_name}"

                # Add a group parser for each class
                if option["group"] not in group_parsers:
                    group_parsers[option["group"]] = parser.add_argument_group(
                        option["group"]
                    )
                group = group_parsers[option["group"]]

                # Add the option itself
                if option["type"] is bool:
                    if "default" in option:
                        action = "store_false" if option["default"] else "store_true"
                    else:
                        action = "store_true"
                    group.add_argument(
                        f"--{option_name}",
                        action=action,
                        help=option["help_msg"],
                    )
                elif option["is_list"]:
                    group.add_argument(
                        f"--{option_name}",
                        nargs="+",
                        type=option["type"],
                        default=option["default"],
                        required=not option["is_optional"],
                        help=option["help_msg"],
                    )
                elif option["is_dict"]:
                    group.add_argument(
                        f"--{option_name}",
                        nargs="+",
                        metavar=f"{option['type'][0]}={option['type'][1]}",
                        default=option["default"],
                        required=not option["is_optional"],
                        help=option["help_msg"],
                    )
                elif option["choices"] is not None:
                    group.add_argument(
                        f"--{option_name}",
                        type=str,
                        choices=option["choices"],
                        default=option["default"],
                        required=not option["is_optional"],
                        help=option["help_msg"],
                    )
                else:
                    group.add_argument(
                        f"--{option_name}",
                        type=option["type"],
                        default=option["default"],
                        required=not option["is_optional"],
                        help=option["help_msg"],
                    )


class _FieldArgs(TypedDict, total=False):
    kw_only: bool


_COMMON_ARGS: _FieldArgs = {"kw_only": True}


def REQUIRED() -> Any:
    return field(**_COMMON_ARGS)


def DEFAULT(default: Any) -> Any:
    if isinstance(default, list):
        return field(default_factory=list, **_COMMON_ARGS)
    if isinstance(default, dict):
        return field(default_factory=dict, **_COMMON_ARGS)
    return field(default=default, **_COMMON_ARGS)


def SYSTEM_DERIVED() -> Any:
    return field(metadata={"argparse": False, "df": False}, **_COMMON_ARGS)


def MODEL_DERIVED() -> Any:
    return field(metadata={"argparse": False}, **_COMMON_ARGS)


def POST_INIT() -> Any:
    return field(init=False, metadata={"argparse": False}, **_COMMON_ARGS)


@dataclass
class TrainConfig(Config):
    tag: str = REQUIRED()
    dataset: MLDataset = REQUIRED()
    dry_run: bool = DEFAULT(False)
    save_model: bool = DEFAULT(False)


@dataclass
class TrainLightningConfig(TrainConfig):
    class Schedulers(StrEnum):
        COSINE = auto()

    profile: bool = DEFAULT(False)
    max_epochs: int = DEFAULT(200)
    patience: int = DEFAULT(3)
    track_memory: str | None = DEFAULT(None)
    lr_scheduler: Schedulers | None = DEFAULT(None)
    monitor_metric: str = DEFAULT("accuracy/validation")
    monitor_mode: str = DEFAULT("max")
