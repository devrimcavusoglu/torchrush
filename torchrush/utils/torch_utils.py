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
