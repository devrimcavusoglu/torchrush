import importlib
import inspect
import io
import re
from collections.abc import Iterable
from pathlib import Path

from datasets import __version__ as _DATASETS_VERSION
from pytorch_lightning import __version__ as _PYTORCH_LIGHTNING_VERSION
from torch import __version__ as _PYTORCH_VERSION


def isiterable(obj):
    return isinstance(obj, Iterable)


def import_module(module_name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_common_init_args(object: object, **kwargs):
    sign = inspect.signature(object.__init__)
    init_args = {k: v for k, v in kwargs.items() if k in sign.parameters}
    return init_args


def get_rush_version():
    version_file = Path(__file__).parent.parent / "__init__.py"
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def get_versions():
    return {
        "datasets": _DATASETS_VERSION,
        "torchrush": get_rush_version(),
        "pytorch": _PYTORCH_VERSION,
        "pytorch-lightning": _PYTORCH_LIGHTNING_VERSION,
    }
