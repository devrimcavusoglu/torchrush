import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple, Union, final

import pytorch_lightning as pl
import torch
from torch.nn import Parameter
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.optim import Optimizer as TorchOptimizer

from torchrush.utils.common import get_versions
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
    r"""
    BaseModule of Rush extending pl.LightningModule. This is the base module one should
    extend for own use cases. You can combine
    """

    def __init__(
        self,
        *args,
        optimizer: Optional[Union[str, TorchOptimizer]] = None,
        criterion: Optional[Union[str, TorchLoss]] = None,
        **kwargs,
    ):
        super(BaseModule, self).__init__()
        self._criterion = None
        self._optimizer = None
        self._criterion_handle, self._optimizer_handle = self.setup_components(criterion, optimizer, **kwargs)
        kwargs = self._clean_kwargs(**kwargs)
        self._rush_config = {
            "optimizer": optimizer if isinstance(optimizer, str) else optimizer.__class__.__name__,
            "criterion": criterion if isinstance(criterion, str) else criterion.__class__.__name__,
            "versions": get_versions(),
            **kwargs,
        }
        self._init_model(*args, **kwargs)

    def _init_model(self, *args, **kwargs):
        pass

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def rush_config(self):
        return self._rush_config

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
        else:
            if not isinstance(criterion, TorchLoss):
                warnings.warn(
                    f"To be automatically constructed `criterion` is expected to be a string or an instance of "
                    f"`torch.nn.modules.loss._Loss`, got `{type(criterion)}`. You need to explicitly define "
                    f"the loss computation logic in `compute_loss()`."
                )
            criterion_args = {}
            criterion_is_object = True

        if isinstance(optimizer, str):
            optimizer_args = get_optimizer_args(optimizer, **kwargs)
            if optimizer_args is None:
                optimizer_args = {}
            optimizer_is_object = False
        else:
            if not isinstance(optimizer, TorchOptimizer):
                warnings.warn(
                    f"To be automatically constructed `optimizer` is expected to be a string or an instance of "
                    f"`torch.optim.Optimizer`, got `{type(optimizer)}`."
                )
            optimizer_args = {}
            optimizer_is_object = True

        return (
            ArgumentHandler(name_or_object=criterion, arguments=criterion_args, is_object=criterion_is_object),
            ArgumentHandler(name_or_object=optimizer, arguments=optimizer_args, is_object=optimizer_is_object),
        )

    @abstractmethod
    def forward(self, x):
        pass

    def compute_loss(self, y_pred, y_true):
        """
        Loss computation logic required for training.
        """
        raise NotImplementedError("`compute_loss` is not implemented, required for training.")

    def params_to_optimize(self) -> Iterator[Parameter]:
        """
        This method determines the parameters to be optimized, which sets up the optimizer in
        a way to update all parameters of the module by default. One must override this method to set up
        optimizer in other use cases (i.e. training/optimizing only few of the layers/layer blocks).

        Returns:
            Returns `torch.nn.Parameter`.
        """
        return self.parameters()

    @final
    def configure_optimizers(self) -> Optional[torch.optim.Optimizer]:
        criterion_handle = self._criterion_handle
        optimizer_handle = self._optimizer_handle
        if criterion_handle.is_object:
            self._criterion = criterion_handle.name_or_object
        else:
            self._criterion = get_criterion_by_name(criterion_handle.name_or_object, **criterion_handle.arguments)
        if optimizer_handle.is_object:
            self._optimizer = optimizer_handle.name_or_object
        else:
            optimizer_handle.arguments["params"] = self.params_to_optimize()
            self._optimizer = get_optimizer_by_name(optimizer_handle.name_or_object, **optimizer_handle.arguments)
        return self.optimizer

    def shared_step(self, batch: Any, batch_idx: int, mode: str = "train") -> Dict[str, Any]:
        """
        Should return dict with key `loss`. Can optionally return keys
        `predictions` and `references` for metric calculation.
        """
        inputs, references = batch
        logits = self(inputs)
        loss = self.compute_loss(logits, references)
        return {"loss": loss, "predictions": torch.argmax(logits, -1), "references": references}

    def _training_step(self, batch, batch_index: int) -> Dict[str, Any]:
        return self.shared_step(batch, batch_index, mode="train")

    @final
    def training_step(self, batch, batch_index: int) -> Dict[str, Any]:
        if self.criterion is None or self.optimizer is None:
            raise AttributeError("'criterion' and 'optimizer' cannot be None for training process.")
        return self._training_step(batch, batch_index)

    def validation_step(self, batch, batch_index):
        return self.shared_step(batch, batch_index, mode="val")

    def test_step(self, batch, batch_index: int) -> Dict[str, Any]:
        return self.shared_step(batch, batch_index, mode="test")

    def log_any(self, any: Dict[str, Any], step: int = None) -> None:
        for logger in self.loggers:
            logger.log_any(any, step)

    def log_hyperparams(self) -> None:
        for logger in self.loggers:
            logger.log_hyperparams(self.rush_config)
