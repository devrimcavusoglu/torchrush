import os
import shutil
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from torchrush.data_loader import DataLoader
from torchrush.dataset import GenericImageClassificationDataset
from torchrush.loggers import CSVLogger, TensorBoardLogger
from torchrush.module.base import BaseModule
from torchrush.module.lenet5 import LeNetForClassification

TEMP_LOG_DIR = "temp_log_dir"


class CallbackForTestingTrainer(Callback):
    losses = []

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: BaseModule, outputs, batch, batch_idx: int
    ) -> None:
        if batch_idx % 39 == 0:
            x, y = batch
            model_out = pl_module(x)
            loss = pl_module.compute_loss(model_out, y)
            self.losses.append(loss.item())


@pytest.fixture
def loss_callback():
    return CallbackForTestingTrainer()


@pytest.fixture(scope="function")
def rush_model():
    pl.seed_everything(42)
    return LeNetForClassification(
        optimizer="SGD", criterion="CrossEntropyLoss", input_size=(28, 28, 1), lr=0.01
    )


@pytest.fixture(scope="function")
def data_loaders():
    train_loader = DataLoader.from_datasets(
        "mnist", split="train", constructor=GenericImageClassificationDataset, batch_size=32
    )
    val_loader = DataLoader.from_datasets(
        "mnist", split="test", constructor=GenericImageClassificationDataset, batch_size=32
    )
    return train_loader, val_loader


def test_rushmodel_logs_with_rushloggers(rush_model, loss_callback):
    SAMPLE_LOG_DICT = {"loss": 0.1, "accuracy": 0.9}

    # create mock loggers
    csv_logger = CSVLogger(save_dir=TEMP_LOG_DIR)
    csv_logger._experiment = MagicMock()

    tb_logger = TensorBoardLogger(save_dir=TEMP_LOG_DIR)
    tb_logger._experiment = MagicMock()

    # set trainer to rush_model
    pl_trainer = pl.Trainer(
        enable_checkpointing=False,
        max_steps=200,
        callbacks=[loss_callback],
        logger=[csv_logger, tb_logger],
    )
    rush_model.trainer = pl_trainer

    # call log_any
    rush_model.log_any(SAMPLE_LOG_DICT)

    # check if the logger is called properly
    csv_logger.experiment.log_metrics.assert_called_with(SAMPLE_LOG_DICT, step=None)
    tb_logger.experiment.add_scalar.assert_called_with("accuracy", 0.9, global_step=None)


def test_pltrainer_trains_with_rushloggers(rush_model, data_loaders, loss_callback):
    csv_logger = CSVLogger(save_dir=TEMP_LOG_DIR)
    tb_logger = TensorBoardLogger(save_dir=TEMP_LOG_DIR)

    train_loader, val_loader = data_loaders
    pl_trainer = pl.Trainer(
        enable_checkpointing=False,
        max_steps=100,
        callbacks=[loss_callback],
        logger=[csv_logger, tb_logger],
    )
    pl_trainer.fit(rush_model, train_loader, val_loader)
    losses = loss_callback.losses

    # remove the temp dir
    shutil.rmtree(TEMP_LOG_DIR)

    assert list(sorted(losses, reverse=True)) == losses
