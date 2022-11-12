from pathlib import Path

from datasets import __version__ as _DATASETS_VERSION
from pytorch_lightning import __version__ as _PYTORCH_LIGHTNING_VERSION
from torch import __version__ as _PYTORCH_VERSION

PROJECT_ROOT = Path(__file__).resolve().parent.parent

__version__ = "0.0.1.dev1"


def get_versions():
    return {
        "datasets": _DATASETS_VERSION,
        "torchrush": __version__,
        "pytorch": _PYTORCH_VERSION,
        "pytorch-lightning": _PYTORCH_LIGHTNING_VERSION,
    }
