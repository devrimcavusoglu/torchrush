import shutil
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from torchrush.data_loader import DataLoader
from torchrush.dataset import GenericImageClassificationDataset
from torchrush.loggers import TensorBoardLogger
from torchrush.metrics import CombinedEvaluations, MetricCallback
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
def rush_model():
    pl.seed_everything(42)
    return LeNetForClassification(
        optimizer="SGD", criterion="CrossEntropyLoss", input_size=(28, 28, 1), lr=0.01
    )


@pytest.fixture
def data_loaders():
    train_loader = DataLoader.from_datasets(
        "mnist", split="train", constructor=GenericImageClassificationDataset, batch_size=32
    )
    val_loader = DataLoader.from_datasets(
        "mnist", split="test", constructor=GenericImageClassificationDataset, batch_size=32
    )
    return train_loader, val_loader


@pytest.fixture
def metric_callback():
    metric_callback = MetricCallback(metrics=["accuracy", "f1", "precision", "recall"])
    assert len(metric_callback.combined_evaluations["train"].metrics) == 4
    assert len(metric_callback.combined_evaluations["val"].metrics) == 4
    assert len(metric_callback.combined_evaluations["val"].metrics) == 4
    return metric_callback


@pytest.fixture
def combined_evaluations():
    # check if metrics are correctly added to the combined evaluations
    combined_evaluations = CombinedEvaluations(metrics=["accuracy", "f1", "precision", "recall"])
    mock_metric = MagicMock()
    combined_evaluations.metrics = [mock_metric]
    combined_evaluations.add_batch([0, 0, 0], [0, 1, 0])
    mock_metric.add_batch.assert_called_with(predictions=[0, 0, 0], references=[0, 1, 0])
    return combined_evaluations


@pytest.fixture
def mock_metric():
    mock_metric = MagicMock()
    return mock_metric


@pytest.fixture
def mock_logger():
    mock_logger = MagicMock()
    return mock_logger


@pytest.fixture
def batch_step_outputs():
    outputs = {
        "loss": torch.tensor(0.5),
        "predictions": torch.tensor([1, 0, 1, 1]),
        "references": torch.tensor([1, 1, 1, 0]),
    }
    return outputs


def prepare_trainer_state(pl_trainer, rush_model, data_loaders, mock_logger):
    # attach dataloaders and model to trainer
    train_loader, val_loader = data_loaders
    pl_trainer._data_connector.attach_data(
        rush_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    pl_trainer.reset_train_dataloader(rush_model)
    pl_trainer.reset_val_dataloader(rush_model)
    pl_trainer.strategy._lightning_module = rush_model

    # check logger is attached to the trainer and model
    assert pl_trainer.loggers == [mock_logger]
    assert rush_model.loggers == [mock_logger]


@pytest.fixture
def pl_trainer_on_train_batch_end(rush_model, mock_logger, data_loaders):
    val_check_interval = 33

    metric_callback = MetricCallback(
        metrics=["accuracy", "f1", "precision", "recall"], log_labelwise_metrics=False
    )
    metric_callback.combined_evaluations = MagicMock()

    pl_trainer = pl.Trainer(
        enable_checkpointing=False,
        max_steps=100,
        val_check_interval=val_check_interval,
        logger=[mock_logger],
        callbacks=[metric_callback],
    )

    prepare_trainer_state(pl_trainer, rush_model, data_loaders, mock_logger)

    # set batch_idx to be the validation batch index
    batch_idx = val_check_interval - 1
    assert (batch_idx + 1) % pl_trainer.val_check_batch == 0
    assert pl_trainer.fit_loop.epoch_loop._should_check_val_fx() == True

    return pl_trainer, batch_idx


@pytest.fixture
def pl_trainer_on_val_batch_with_mock_combined_evaluations(rush_model, mock_logger, data_loaders):
    val_check_interval = 33

    metric_callback = MetricCallback(
        metrics=["accuracy", "f1", "precision", "recall"], log_labelwise_metrics=False
    )
    metric_callback.combined_evaluations = MagicMock()

    pl_trainer = pl.Trainer(
        enable_checkpointing=False,
        max_steps=100,
        val_check_interval=val_check_interval,
        logger=[mock_logger],
        callbacks=[metric_callback],
    )

    prepare_trainer_state(pl_trainer, rush_model, data_loaders, mock_logger)

    # set batch_idx to be the validation batch index
    batch_idx = val_check_interval - 1
    assert (batch_idx + 1) % pl_trainer.val_check_batch == 0
    assert pl_trainer.fit_loop.epoch_loop._should_check_val_fx() == True

    return pl_trainer, batch_idx, metric_callback


@pytest.fixture
def pl_trainer_on_val_batch_with_mock_metrics(rush_model, mock_logger, data_loaders):
    val_check_interval = 33

    metric_callback = MetricCallback(
        metrics=["accuracy", "f1", "precision", "recall"], log_labelwise_metrics=False
    )
    metric_callback.combined_evaluations["val"].metrics = [MagicMock(), MagicMock()]
    metric_callback.combined_evaluations["val"].metrics[0].name = "accuracy"
    metric_callback.combined_evaluations["val"].metrics[1].name = "f1"

    pl_trainer = pl.Trainer(
        enable_checkpointing=False,
        max_steps=100,
        val_check_interval=val_check_interval,
        logger=[mock_logger],
        callbacks=[metric_callback],
    )

    prepare_trainer_state(pl_trainer, rush_model, data_loaders, mock_logger)

    # set batch_idx to be the validation batch index
    batch_idx = val_check_interval - 1
    assert (batch_idx + 1) % pl_trainer.val_check_batch == 0

    return pl_trainer, batch_idx, metric_callback


@pytest.fixture
def pl_trainer_on_val_batch_with_labelwise_mock_metrics(rush_model, mock_logger, data_loaders):
    val_check_interval = 33

    metric_callback = MetricCallback(
        metrics=["accuracy", "f1", "precision", "recall"], log_labelwise_metrics=True, labels=[0, 1]
    )
    metric_callback.combined_evaluations["test"].metrics = [MagicMock(), MagicMock()]
    metric_callback.combined_evaluations["test"].metrics[0].name = "accuracy"
    metric_callback.combined_evaluations["test"].metrics[1].name = "f1"

    pl_trainer = pl.Trainer(
        enable_checkpointing=False,
        max_steps=100,
        val_check_interval=val_check_interval,
        logger=[mock_logger],
        callbacks=[metric_callback],
    )

    prepare_trainer_state(pl_trainer, rush_model, data_loaders, mock_logger)

    # set batch_idx to be the validation batch index
    batch_idx = val_check_interval - 1
    assert (batch_idx + 1) % pl_trainer.val_check_batch == 0

    return pl_trainer, batch_idx, metric_callback


@pytest.fixture
def pl_trainer_on_val_epoch_with_labelwise_metric_callback(rush_model, mock_logger, data_loaders):
    check_val_every_n_epoch = 1

    metric_callback = MetricCallback(
        metrics=["accuracy", "f1", "precision", "recall"], log_labelwise_metrics=True, labels=[0, 1]
    )

    pl_trainer = pl.Trainer(
        enable_checkpointing=False,
        max_steps=100,
        check_val_every_n_epoch=check_val_every_n_epoch,
        logger=[mock_logger],
        callbacks=[metric_callback],
    )

    prepare_trainer_state(pl_trainer, rush_model, data_loaders, mock_logger)

    # set current_epoch to 1
    pl_trainer.fit_loop.epoch_progress.current.completed = 1
    assert pl_trainer.current_epoch == 1

    # set to last batch of epoch: 1874
    batch_idx = max(1, int(pl_trainer.num_training_batches * pl_trainer.val_check_interval)) - 1
    assert (batch_idx + 1) % pl_trainer.val_check_batch == 0

    return pl_trainer, batch_idx, metric_callback


def test_rushmetrics_accuracy(combined_evaluations, mock_metric):
    mock_metric = MagicMock()

    # check if accuracy is correctly computed in the combined evaluations
    mock_metric.name = "accuracy"
    combined_evaluations.metrics = [mock_metric]
    combined_evaluations.compute()
    mock_metric.compute.assert_called_with()
    combined_evaluations.compute(labelwise=True)
    mock_metric.compute.assert_called_with()


def test_rushmetrics_precision(combined_evaluations, mock_metric):
    # check if precision is correctly computed in the combined evaluations
    mock_metric.name = "precision"
    combined_evaluations.metrics = [mock_metric]
    combined_evaluations.compute()
    mock_metric.compute.assert_called_with(average="macro", zero_division=0)
    combined_evaluations.compute(labelwise=True)
    mock_metric.compute.assert_called_with(average=None, zero_division=0)


def test_rushmetrics_recall(combined_evaluations, mock_metric):
    mock_metric = MagicMock()

    # check if recall is correctly computed in the combined evaluations
    mock_metric.name = "recall"
    combined_evaluations.metrics = [mock_metric]
    combined_evaluations.compute()
    mock_metric.compute.assert_called_with(average="macro")
    combined_evaluations.compute(labelwise=True)
    mock_metric.compute.assert_called_with(average=None)


def test_rushmetrics_callback_train_batch_end(
    rush_model, mock_logger, batch_step_outputs, pl_trainer_on_val_batch_with_mock_combined_evaluations
):
    # scenario 1: batch_end + train
    pl_trainer, batch_idx, metric_callback = pl_trainer_on_val_batch_with_mock_combined_evaluations

    metric_callback.on_train_batch_end(pl_trainer, rush_model, batch_step_outputs, None, batch_idx)
    metric_callback.combined_evaluations["train"].add_batch.assert_called_with(
        predictions=batch_step_outputs["predictions"], references=batch_step_outputs["references"]
    )
    metric_callback.combined_evaluations["train"].compute.assert_called_with(labelwise=False)
    mock_logger.log_any.assert_called_with({"train/loss": batch_step_outputs["loss"]}, batch_idx)


def test_rushmetrics_callback_val_steps(
    rush_model, mock_logger, batch_step_outputs, pl_trainer_on_val_batch_with_mock_metrics
):
    # scenario 2: batch_end + val
    pl_trainer, batch_idx, metric_callback = pl_trainer_on_val_batch_with_mock_metrics

    metric_callback.on_validation_batch_end(
        pl_trainer, rush_model, batch_step_outputs, None, batch_idx, 0
    )
    metric_callback.combined_evaluations["val"].metrics[0].add_batch.assert_called()
    metric_callback.on_validation_end(pl_trainer, rush_model)
    metric_callback.combined_evaluations["val"].metrics[0].compute.assert_called_with()
    metric_callback.combined_evaluations["val"].metrics[1].compute.assert_called_with(average="macro")
    mock_logger.log_any.assert_called_with({"val/loss": batch_step_outputs["loss"]}, 0)


def test_rushmetrics_callback_test_batch_end_labelwise(
    rush_model, mock_logger, batch_step_outputs, pl_trainer_on_val_batch_with_labelwise_mock_metrics
):
    # scenario 3: labelwise + batch_end + test
    pl_trainer, batch_idx, metric_callback = pl_trainer_on_val_batch_with_labelwise_mock_metrics

    metric_callback.on_test_batch_end(pl_trainer, rush_model, batch_step_outputs, None, batch_idx, 0)
    metric_callback.combined_evaluations["test"].metrics[0].add_batch.assert_called()

    metric_callback.on_test_end(pl_trainer, rush_model)
    metric_callback.combined_evaluations["test"].metrics[0].compute.assert_called_with()
    metric_callback.combined_evaluations["test"].metrics[1].compute.assert_called_with(average=None)
    mock_logger.log_any.assert_called_with({"test/loss": batch_step_outputs["loss"]}, 0)


def test_rushmetrics_callback_val_epoch_end_labelwise(
    rush_model, mock_logger, batch_step_outputs, pl_trainer_on_val_epoch_with_labelwise_metric_callback
):
    # scenario 4: labelwise + epoch_end + validation
    pl_trainer, batch_idx, metric_callback = pl_trainer_on_val_epoch_with_labelwise_metric_callback

    metric_callback.on_validation_batch_end(
        pl_trainer, rush_model, batch_step_outputs, None, batch_idx, 0
    )
    metric_callback.on_validation_end(pl_trainer, rush_model)
    metric_callback.on_validation_epoch_end(pl_trainer, rush_model)
    mock_logger.log_any.assert_called_with({"val/recall_1": 0.6666666666666666}, 0)


def test_pltrainer_trains_with_rushmetrics(rush_model, data_loaders):
    tb_logger = TensorBoardLogger(save_dir=TEMP_LOG_DIR)

    metric_callback = MetricCallback(metrics=["accuracy", "f1", "precision", "recall"])

    train_loader, val_loader = data_loaders
    pl_trainer = pl.Trainer(
        enable_checkpointing=False, max_steps=20, logger=tb_logger, callbacks=[metric_callback]
    )
    pl_trainer.fit(rush_model, train_loader, val_loader)

    # remove the temp dir
    shutil.rmtree(TEMP_LOG_DIR)


def test_pltrainer_trains_with_rushmetrics_labelwise(rush_model, data_loaders):
    tb_logger = TensorBoardLogger(save_dir=TEMP_LOG_DIR)

    train_loader, val_loader = data_loaders

    metric_callback = MetricCallback(
        metrics=["accuracy", "f1", "precision", "recall"],
        log_labelwise_metrics=True,
        labels=train_loader.dataset._dataset.features["label"].names,
    )

    pl_trainer = pl.Trainer(
        enable_checkpointing=False, max_steps=20, logger=tb_logger, callbacks=[metric_callback]
    )
    pl_trainer.fit(rush_model, train_loader, val_loader)

    # remove the temp dir
    shutil.rmtree(TEMP_LOG_DIR)
