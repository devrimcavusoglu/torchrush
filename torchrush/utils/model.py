from pathlib import Path

import torch

from torchrush.utils.common import import_module


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def load_model(module_path: str, model_name: str, **kwargs) -> torch.nn.Module:
    module_path = Path(module_path)
    model_module = import_module(module_path.stem, str(module_path))
    model_klass = getattr(model_module, model_name)
    return model_klass(**kwargs)
