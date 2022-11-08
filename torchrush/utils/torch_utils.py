import random

import numpy as np
import torch
from torch import optim
from torch.nn.modules import loss

from torchrush.utils.common import get_common_init_args


def get_optimizer_args(optimizer, **kwargs):
    optimizer_klass = getattr(optim, optimizer)
    return get_common_init_args(optimizer_klass, **kwargs)


def get_criterion_args(criterion, **kwargs):
    criterion_klass = getattr(loss, criterion)
    return get_common_init_args(criterion_klass, **kwargs)


def get_optimizer_by_name(optimizer: str, **kwargs):
    optimizer_klass = getattr(optim, optimizer)
    return optimizer_klass(**kwargs)


def get_criterion_by_name(criterion: str, **kwargs):
    criterion_klass = getattr(loss, criterion)
    return criterion_klass(**kwargs)


def seed_all(seed: int):
    # Seeding all RNG, taken from HF.transformers.
    # ref: https://github.com/huggingface/transformers/blob/94b3f544a1f5e04b78d87a2ae32a7ac252e22e31/src/transformers/trainer_utils.py#L83
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
