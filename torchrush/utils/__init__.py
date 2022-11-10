from datasets import __version__ as _DATASETS_VERSION
from pytorch_lightning import __version__ as _PYTORCH_LIGHTNING_VERSION
from torch import __version__ as _PYTORCH_VERSION

from torchrush import __version__ as _RUSH_VERSION


def get_versions():
    return {
        "datasets": _DATASETS_VERSION,
        "torchrush": _RUSH_VERSION,
        "pytorch": _PYTORCH_VERSION,
        "pyotorch-lightning": _PYTORCH_LIGHTNING_VERSION,
    }
