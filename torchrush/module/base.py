import json
import logging
import os
import shutil
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union, final

import pytorch_lightning as pl
import requests
import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
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

RUSH_WEIGHTS_NAME = "rushmodel.bin"
RUSH_CONFIG_NAME = "rushconfig.json"
RUSH_FILE_NAME = "rush.py"

logger = logging.getLogger(__name__)


@dataclass
class ArgumentHandler:
    name_or_object: Union[str, Any]
    arguments: Dict[str, Any] = None
    is_object: bool = False

    def __post_init__(self):
        # in case of None set empty str
        self.name_or_object = self.name_or_object or ""


class BaseModule(pl.LightningModule, PyTorchModelHubMixin):
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
        self._criterion_handle, self._optimizer_handle = self.setup_components(
            criterion, optimizer, **kwargs
        )
        kwargs = self._clean_kwargs(**kwargs)

        self._rush_config = self.__create_rush_config(**kwargs)

        self._init_model(*args, **kwargs)

    def __create_rush_config(self, **kwargs) -> Dict[str, Any]:
        cfg = {"versions": get_versions(), "model": self.__class__.__name__, **kwargs}
        handlers = [("criterion", self._criterion_handle), ("optimizer", self._optimizer_handle)]
        for handler, handle in handlers:
            if handle.is_object:
                name = handle.name_or_object.__class__.__name__
            else:
                name = handle.name_or_object

            args = {"name": name, **handle.arguments}
            cfg.update({handler: args})
        return cfg

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
            ArgumentHandler(
                name_or_object=criterion, arguments=criterion_args, is_object=criterion_is_object
            ),
            ArgumentHandler(
                name_or_object=optimizer, arguments=optimizer_args, is_object=optimizer_is_object
            ),
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
            if criterion_handle.name_or_object == "":
                raise ValueError("'criterion' is not defined.")
            self._criterion = criterion_handle.name_or_object
        else:
            self._criterion = get_criterion_by_name(
                criterion_handle.name_or_object, **criterion_handle.arguments
            )
        if optimizer_handle.is_object:
            if optimizer_handle.name_or_object == "":
                raise ValueError("'optimizer' is not defined.")
            self._optimizer = optimizer_handle.name_or_object
        else:
            optimizer_handle.arguments["params"] = self.params_to_optimize()
            self._optimizer = get_optimizer_by_name(
                optimizer_handle.name_or_object, **optimizer_handle.arguments
            )
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

    def _save_pretrained(self, save_directory):
        """
        Overwrite this method if you wish to save specific layers instead of the
        complete model.
        """
        import torch

        # saving model
        path = os.path.join(save_directory, RUSH_WEIGHTS_NAME)
        torch.save(self.state_dict(), path)

        # saving config
        path = os.path.join(save_directory, RUSH_CONFIG_NAME)
        with open(path, "w") as f:
            json.dump(self._rush_config, f)

        # saving rush file
        import importlib

        subclass = importlib.import_module(self.__module__)
        subclass_file_path = os.path.realpath(subclass.__file__)
        target_file_path = os.path.join(save_directory, RUSH_FILE_NAME)
        shutil.copy(subclass_file_path, target_file_path)

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        token,
        map_location="cpu",
        strict=False,
        **model_kwargs,
    ):
        """
        Overwrite this method to initialize your model in a different way.
        """
        import os

        import torch
        from huggingface_hub import hf_hub_download

        map_location = torch.device(map_location)

        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, RUSH_WEIGHTS_NAME)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=RUSH_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        model = cls(**model_kwargs)

        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()

        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **model_kwargs,
    ):
        r"""
        Download and instantiate a model from the Hugging Face Hub.

                Parameters:
                    pretrained_model_name_or_path (`str` or `os.PathLike`):
                        Can be either:
                            - A string, the `model id` of a pretrained model
                              hosted inside a model repo on huggingface.co.
                              Valid model ids can be located at the root-level,
                              like `bert-base-uncased`, or namespaced under a
                              user or organization name, like
                              `dbmdz/bert-base-german-cased`.
                            - You can add `revision` by appending `@` at the end
                              of model_id simply like this:
                              `dbmdz/bert-base-german-cased@main` Revision is
                              the specific model version to use. It can be a
                              branch name, a tag name, or a commit id, since we
                              use a git-based system for storing models and
                              other artifacts on huggingface.co, so `revision`
                              can be any identifier allowed by git.
                            - A path to a `directory` containing model weights
                              saved using
                              [`~transformers.PreTrainedModel.save_pretrained`],
                              e.g., `./my_model_directory/`.
                            - `None` if you are both providing the configuration
                              and state dictionary (resp. with keyword arguments
                              `config` and `state_dict`).
                    force_download (`bool`, *optional*, defaults to `False`):
                        Whether to force the (re-)download of the model weights
                        and configuration files, overriding the cached versions
                        if they exist.
                    resume_download (`bool`, *optional*, defaults to `False`):
                        Whether to delete incompletely received files. Will
                        attempt to resume the download if such a file exists.
                    proxies (`Dict[str, str]`, *optional*):
                        A dictionary of proxy servers to use by protocol or
                        endpoint, e.g., `{'http': 'foo.bar:3128',
                        'http://hostname': 'foo.bar:4012'}`. The proxies are
                        used on each request.
                    token (`str` or `bool`, *optional*):
                        The token to use as HTTP bearer authorization for remote
                        files. If `True`, will use the token generated when
                        running `transformers-cli login` (stored in
                        `~/.huggingface`).
                    cache_dir (`Union[str, os.PathLike]`, *optional*):
                        Path to a directory in which a downloaded pretrained
                        model configuration should be cached if the standard
                        cache should not be used.
                    local_files_only(`bool`, *optional*, defaults to `False`):
                        Whether to only look at local files (i.e., do not try to
                        download the model).
                    model_kwargs (`Dict`, *optional*):
                        model_kwargs will be passed to the model during
                        initialization

                <Tip>

                Passing `token=True` is required when you want to use a
                private model.

                </Tip>
        """

        model_id = pretrained_model_name_or_path

        revision = None
        if len(model_id.split("@")) == 2:
            model_id, revision = model_id.split("@")

        config_file: Optional[str] = None
        if os.path.isdir(model_id):
            if RUSH_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, RUSH_CONFIG_NAME)
            else:
                logger.warning(f"{RUSH_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=RUSH_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except requests.exceptions.RequestException:
                logger.warning(f"{RUSH_CONFIG_NAME} not found in HuggingFace Hub")

        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            # remove the keys that are not model kwargs
            config.pop("versions", None)
            config.pop("model", None)
            model_kwargs.update(config)

        return cls._from_pretrained(
            model_id,
            revision,
            cache_dir,
            force_download,
            proxies,
            resume_download,
            local_files_only,
            token,
            **model_kwargs,
        )
