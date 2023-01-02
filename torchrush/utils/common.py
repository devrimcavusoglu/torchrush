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
    """Load a module from a given Python file path and module name.

    Parameters:
        module_name (str): The name of the module to be loaded.
        filepath (str): The path to the Python file containing the module.

    Returns:
        type: The class object.
    """
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_class(class_name: str, filepath: str):
    """Load a class from a given Python file path and class name.

    Parameters:
        class_name (str): The name of the class to be loaded.
        filepath (str): The path to the Python file containing the class.

    Returns:
        type: The class object.
    """
    # Load the module from the file path
    spec = importlib.util.spec_from_file_location(class_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class from the module
    return getattr(module, class_name)


def get_common_init_args(object: object, **kwargs) -> dict:
    """Get the arguments common to the given object's `__init__` method and the provided keyword arguments.

    Parameters:
        object (object): The object whose `__init__` method will be inspected.
        **kwargs (dict): The keyword arguments to check for commonality with the `__init__` method's arguments.

    Returns:
        dict: A dictionary of the common arguments, with the argument names as keys and the provided values as values.
    """
    # Get the signature of the object's __init__ method
    sign = inspect.signature(object.__init__)

    # Filter the provided keyword arguments to only include those that are present in the __init__ method's signature
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
