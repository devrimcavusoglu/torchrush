import importlib
import inspect
from collections.abc import Iterable


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
