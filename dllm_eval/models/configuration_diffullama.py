"""
Diffullama configuration
"""
from transformers import AutoConfig, PretrainedConfig

from enum import Enum
from os import PathLike
from typing import Union
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
