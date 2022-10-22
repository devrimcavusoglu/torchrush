from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union, Dict, Any

import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.optim import Optimizer as TorchOptimizer

from torchrush.utils.torch_utils import get_optimizer_by_name, get_criterion_by_name, get_optimizer_args, \
    get_criterion_args


@dataclass
class ArgumentHandler:
    name_or_object: Union[str, Any]
    arguments: Dict[str, Any] = None
    is_object: bool = False


class BaseModule(pl.LightningModule):
    def __init__(self,
                 criterion: Union[str, TorchLoss],
                 optimizer: Union[str, TorchOptimizer],
                 **kwargs
    ):
        super(BaseModule, self).__init__()
        self._criterion = None
        self._optimizer = None
        self._creterion_handle, self._optimizer_handle = self.setup_components(criterion, optimizer, **kwargs)

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    def setup_components(
            self,
            criterion: Union[str, TorchLoss],
            optimizer: Union[str, TorchOptimizer],
            **kwargs
    ) -> Tuple[ArgumentHandler, ArgumentHandler]:
        if isinstance(criterion, str):
            criterion_args = get_criterion_args(criterion, **kwargs)
            if criterion_args is None:
                criterion_args = {}
            criterion_is_object = False
        elif not isinstance(criterion, TorchLoss):
            raise ValueError(f"Expecting `str` or `torch.nn.modules._Loss` object, got "
                             f"`{type(criterion)}`.")
        else:
            criterion_args = {}
            criterion_is_object = True

        if isinstance(optimizer, str):
            optimizer_args = get_optimizer_args(optimizer, **kwargs)
            if optimizer_args is None:
                optimizer_args = {}
            optimizer_is_object = False
        elif not isinstance(criterion, TorchOptimizer):
            raise ValueError(f"Expecting `str` or `torch.optim.Optimizer` object, got "
                             f"`{type(optimizer)}`.")
        else:
            optimizer_args = {}
            optimizer_is_object = True

        return (
            ArgumentHandler(name_or_object=criterion, arguments=criterion_args, is_object=criterion_is_object),
            ArgumentHandler(name_or_object=optimizer, arguments=optimizer_args, is_object=optimizer_is_object)
        )

    @abstractmethod
    def _forward(self, x):
        pass

    def forward(self, x):
        x = self.preprocess(x)
        x = self._forward(x)
        x = self.postprocess(x)
        return x

    @abstractmethod
    def compute_loss(self, y_pred, y_true):
        pass

    def configure_optimizers(self):
        criterion_handle = self._creterion_handle
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
        loss = self.compute_loss(model_out, y)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        model_out = self(x)
        loss = self.compute_loss(model_out, y)
        return loss

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class BaseMLP(BaseModule):
    def __init__(
            self,
            input_size: Tuple,
            output_size: int,
            criterion: Union[str, TorchLoss],
            optimizer: Union[str, TorchOptimizer],
            *args,
            **kwargs
    ):
        super(BaseMLP, self).__init__(criterion, optimizer, **kwargs)
        self.user_input_size = input_size
        if len(input_size) == 3:
            channel_size, input_width, input_height = input_size
        elif len(input_size) == 2:
            input_width, input_height = input_size
            channel_size = 1
        elif len(input_size) == 1:
            input_width, input_height = input_size, input_size
            channel_size = 1
        else:
            raise ValueError("Unsupported input type")

        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = channel_size
        self.input_nodes = input_width * input_height * channel_size
        self.input_size = (1, self.input_nodes)
        self.output_size = output_size
        self._init(*args, **kwargs)

    @abstractmethod
    def _init(self, *args, **kwargs):
        pass

    def preprocess(self, x):
        return x.view(x.size(0), -1)

    def compute_loss(self, y_pred, y_true):
        if y_true.ndim == 1:
            y_true = F.one_hot(y_true, self.output_size) * 1.0
        return self.criterion(y_pred, y_true)


class BaseConvNet(BaseModule):
    def __init__(
            self,
            input_size: Tuple,
            criterion: Union[str, TorchLoss],
            optimizer: Union[str, TorchOptimizer],
            *args,
            **kwargs
    ):
        super(BaseConvNet, self).__init__(criterion, optimizer, *args, **kwargs)
        self.input_size = input_size
        if len(input_size) == 3:
            channel_size, input_width, input_height = input_size
        elif len(input_size) == 2:
            input_width, input_height = input_size
            channel_size = 1
        elif len(input_size) == 1:
            input_width, input_height = input_size, input_size
            channel_size = 1
        else:
            raise ValueError("Unsupported input type")

        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = channel_size
        self.input_nodes = channel_size, input_width, input_height
        self._init(*args, **kwargs)

    @abstractmethod
    def _init(self, *args, **kwargs):
        pass

