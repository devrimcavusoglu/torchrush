from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import pytorch_lightning as pl
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.optim import Optimizer as TorchOptimizer

from torchrush.utils.torch_utils import (
    get_criterion_args,
    get_criterion_by_name,
    get_optimizer_args,
    get_optimizer_by_name,
)


@dataclass
class ArgumentHandler:
    name_or_object: Union[str, Any]
    arguments: Dict[str, Any] = None
    is_object: bool = False


class BaseModule(pl.LightningModule):
    def __init__(self, criterion: Union[str, TorchLoss], optimizer: Union[str, TorchOptimizer], *args, **kwargs):
        super(BaseModule, self).__init__()
        self._criterion = None
        self._optimizer = None
        self._criterion_handle, self._optimizer_handle = self.setup_components(criterion, optimizer, **kwargs)
        kwargs = self._clean_kwargs(**kwargs)
        self._init(*args, **kwargs)

    @abstractmethod
    def _init(self, *args, **kwargs):
        pass

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    def _clean_kwargs(self, **kwargs) -> Dict[str, Any]:
        keys_removed = []
        for k, v in kwargs.items():
            if k in self._criterion_handle.arguments or k in self._optimizer_handle.arguments:
                keys_removed.append(k)
        for key in keys_removed:
            kwargs.pop(key)
        return kwargs

    def setup_components(
        self, criterion: Union[str, TorchLoss], optimizer: Union[str, TorchOptimizer], **kwargs
    ) -> Tuple[ArgumentHandler, ArgumentHandler]:
        if isinstance(criterion, str):
            criterion_args = get_criterion_args(criterion, **kwargs)
            if criterion_args is None:
                criterion_args = {}
            criterion_is_object = False
        elif not isinstance(criterion, TorchLoss):
            raise ValueError(f"Expecting `str` or `torch.nn.modules._Loss` object, got " f"`{type(criterion)}`.")
        else:
            criterion_args = {}
            criterion_is_object = True

        if isinstance(optimizer, str):
            optimizer_args = get_optimizer_args(optimizer, **kwargs)
            if optimizer_args is None:
                optimizer_args = {}
            optimizer_is_object = False
        elif not isinstance(criterion, TorchOptimizer):
            raise ValueError(f"Expecting `str` or `torch.optim.Optimizer` object, got " f"`{type(optimizer)}`.")
        else:
            optimizer_args = {}
            optimizer_is_object = True

        return (
            ArgumentHandler(name_or_object=criterion, arguments=criterion_args, is_object=criterion_is_object),
            ArgumentHandler(name_or_object=optimizer, arguments=optimizer_args, is_object=optimizer_is_object),
        )

    @abstractmethod
    def compute_loss(self, y_pred, y_true):
        pass

    @abstractmethod
    def _forward(self, x):
        pass

    def forward(self, x):
        x = self.preprocess(x)
        x = self._forward(x)
        x = self.postprocess(x)
        return x

    def configure_optimizers(self):
        criterion_handle = self._criterion_handle
        optimizer_handle = self._optimizer_handle
        if criterion_handle.is_object:
            self._criterion = criterion_handle.name_or_object
        else:
            self._criterion = get_criterion_by_name(criterion_handle.name_or_object, **criterion_handle.arguments)
        if optimizer_handle.is_object:
            self._optimizer = optimizer_handle.name_or_object
        else:
            optimizer_handle.arguments["params"] = self.parameters()
            self._optimizer = get_optimizer_by_name(optimizer_handle.name_or_object, **optimizer_handle.arguments)
        return self.optimizer

    def training_step(self, batch, batch_index):
        x, y = batch
        model_out = self(x)
        return self.compute_loss(model_out, y)

    def validation_step(self, batch, batch_index):
        x, y = batch
        model_out = self(x)
        return self.compute_loss(model_out, y)

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x
