import shutil
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from torchrush.data_loader import DataLoader
from torchrush.dataset import GenericImageClassificationDataset
from torchrush.loggers import TensorBoardLogger
from torchrush.metrics import CombinedEvaluations, MetricCallback
from torchrush.model.base import BaseModule
from torchrush.model.lenet5 import LeNetForClassification
import torch

TEMP_LOG_DIR = "temp_log_dir"


class CallbackForTestingTrainer(Callback):
    losses = []

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: BaseModule, outputs, batch, batch_idx: int) -> None:
        if batch_idx % 39 == 0:
            x, y = batch
            model_out = pl_module(x)
            loss = pl_module.compute_loss(model_out, y)
            self.losses.append(loss.item())


def get_rush_model():
    pl.seed_everything(42)
    return LeNetForClassification(optimizer="SGD", criterion="CrossEntropyLoss", input_size=(28, 28, 1), lr=0.01)


def get_data_loaders():
    train_loader = DataLoader.from_datasets(
        "mnist", split="train", constructor=GenericImageClassificationDataset, batch_size=32
    )
    val_loader = DataLoader.from_datasets(
        "mnist", split="test", constructor=GenericImageClassificationDataset, batch_size=32
    )
    return train_loader, val_loader


def test_rushmetrics():
    metric_callback = MetricCallback(metrics=["accuracy", "f1", "precision", "recall"], log_on="batch_end")
    assert len(metric_callback.combined_evaluations["train"].metrics) == 4
    assert len(metric_callback.combined_evaluations["val"].metrics) == 4
    assert len(metric_callback.combined_evaluations["val"].metrics) == 4

    # check if metrics are correctly added to the combined evaluations
    combined_evaluations = CombinedEvaluations(metrics=["accuracy"])
    mock_metric = MagicMock()
    combined_evaluations.metrics = [mock_metric]
    combined_evaluations.add_batch([0, 0, 0], [0, 1, 0])

    mock_metric.add_batch.assert_called_with(predictions=[0, 0, 0], references=[0, 1, 0])

    # check if metrics are correctly computed in the combined evaluations
    mock_metric.name = "accuracy"
    combined_evaluations.metrics = [mock_metric]
    combined_evaluations.compute()
    mock_metric.compute.assert_called_with()
    combined_evaluations.compute(labelwise=True)
    mock_metric.compute.assert_called_with()

    mock_metric.name = "precision"
    combined_evaluations.metrics = [mock_metric]
    combined_evaluations.compute()
    mock_metric.compute.assert_called_with(average="macro", zero_division=0)
    combined_evaluations.compute(labelwise=True)
    mock_metric.compute.assert_called_with(average=None, zero_division=0)

    mock_metric.name = "recall"
    combined_evaluations.metrics = [mock_metric]
    combined_evaluations.compute()
    mock_metric.compute.assert_called_with(average="macro")
    combined_evaluations.compute(labelwise=True)
    mock_metric.compute.assert_called_with(average=None)


def test_rushmetrics_callback():
    tb_logger = MagicMock()
    rush_model = get_rush_model()

    outputs = {
        "loss": torch.tensor(0.5),
        "predictions": torch.tensor([1, 0, 1, 1]),
        "references": torch.tensor([1, 1, 1, 0]),
    }

    # scenario 1: batch_end + train
    metric_callback = MetricCallback(
        metrics=["accuracy", "f1", "precision", "recall"], log_on="batch_end", log_labelwise_metrics=False
    )
    metric_callback.combined_evaluations = MagicMock()

    pl_trainer = pl.Trainer(enable_checkpointing=False, max_steps=20, logger=[tb_logger], callbacks=[metric_callback])
    assert pl_trainer.loggers == [tb_logger]
    rush_model.trainer = pl_trainer
    assert rush_model.loggers == [tb_logger]

    batch_idx = 33
    assert batch_idx % metric_callback.eval_freq == 0
    metric_callback.on_train_batch_end(pl_trainer, rush_model, outputs, None, batch_idx)
    metric_callback.combined_evaluations["train"].add_batch.assert_called_with(
        predictions=outputs["predictions"], references=outputs["references"]
    )
    metric_callback.combined_evaluations["train"].compute.assert_called_with(labelwise=False)
    tb_logger.log_any.assert_called_with({"train/loss": outputs["loss"].item()}, batch_idx)

    # scenario 2: batch_end + val
    metric_callback = MetricCallback(
        metrics=["accuracy", "f1", "precision", "recall"], log_on="batch_end", log_labelwise_metrics=False
    )
    metric_callback.combined_evaluations["val"].metrics = [MagicMock(), MagicMock()]
    metric_callback.combined_evaluations["val"].metrics[0].name = "accuracy"
    metric_callback.combined_evaluations["val"].metrics[1].name = "f1"

    pl_trainer = pl.Trainer(enable_checkpointing=False, max_steps=20, logger=[tb_logger], callbacks=[metric_callback])
    assert pl_trainer.loggers == [tb_logger]
    rush_model.trainer = pl_trainer
    assert rush_model.loggers == [tb_logger]

    batch_idx = 35
    assert batch_idx % metric_callback.eval_freq == 0
    metric_callback.on_validation_batch_end(pl_trainer, rush_model, outputs, None, batch_idx, 0)
    metric_callback.combined_evaluations["val"].metrics[0].compute.assert_called_with()
    metric_callback.combined_evaluations["val"].metrics[1].compute.assert_called_with(average="macro")
    tb_logger.log_any.assert_called_with({"val/loss": outputs["loss"].item()}, batch_idx)

    # scenario 3: labelwise + batch_end + test
    metric_callback = MetricCallback(
        metrics=["accuracy", "f1", "precision", "recall"], log_on="batch_end", log_labelwise_metrics=True, labels=[0, 1]
    )
    metric_callback.combined_evaluations["test"].metrics = [MagicMock(), MagicMock()]
    metric_callback.combined_evaluations["test"].metrics[0].name = "accuracy"
    metric_callback.combined_evaluations["test"].metrics[1].name = "f1"

    pl_trainer = pl.Trainer(enable_checkpointing=False, max_steps=20, logger=[tb_logger], callbacks=[metric_callback])
    assert pl_trainer.loggers == [tb_logger]
    rush_model.trainer = pl_trainer
    assert rush_model.loggers == [tb_logger]

    batch_idx = 37
    assert batch_idx % metric_callback.eval_freq == 0
    metric_callback.on_test_batch_end(pl_trainer, rush_model, outputs, None, batch_idx, 0)
    metric_callback.combined_evaluations["test"].metrics[0].compute.assert_called_with()
    metric_callback.combined_evaluations["test"].metrics[1].compute.assert_called_with(average=None)
    tb_logger.log_any.assert_called_with({"test/loss": outputs["loss"].item()}, batch_idx)

    # scenario 4: labelwise + epoch_end + validation
    metric_callback = MetricCallback(
        metrics=["accuracy", "f1", "precision", "recall"], log_on="epoch_end", log_labelwise_metrics=True, labels=[0, 1]
    )

    pl_trainer = pl.Trainer(enable_checkpointing=False, max_steps=20, logger=[tb_logger], callbacks=[metric_callback])
    assert pl_trainer.loggers == [tb_logger]
    pl_trainer.fit_loop.epoch_progress.current.completed = 1  # set current_epoch to 1
    assert pl_trainer.current_epoch == 1
    assert pl_trainer.current_epoch % metric_callback.eval_freq == 0
    rush_model.trainer = pl_trainer
    assert rush_model.loggers == [tb_logger]

    batch_idx = 39
    metric_callback.on_validation_batch_end(pl_trainer, rush_model, outputs, None, batch_idx, 0)
    metric_callback.on_validation_epoch_end(pl_trainer, rush_model)
    tb_logger.log_any.assert_called_with({"val/recall_1": 0.6666666666666666}, 1)


def test_pltrainer_trains_with_rushmetrics():
    # exp 1
    metric_callback = MetricCallback(metrics=["accuracy", "f1", "precision", "recall"], log_on="batch_end")

    tb_logger = TensorBoardLogger(save_dir=TEMP_LOG_DIR)

    train_loader, val_loader = get_data_loaders()
    rush_model = get_rush_model()
    pl_trainer = pl.Trainer(enable_checkpointing=False, max_steps=20, logger=tb_logger, callbacks=[metric_callback])
    pl_trainer.fit(rush_model, train_loader, val_loader)

    # exp 2
    metric_callback = MetricCallback(
        metrics=["accuracy", "f1", "precision", "recall"],
        log_on="batch_end",
        log_labelwise_metrics=True,
        labels=train_loader.dataset._dataset.features["label"].names,
    )

    pl_trainer = pl.Trainer(enable_checkpointing=False, max_steps=20, logger=tb_logger, callbacks=[metric_callback])
    pl_trainer.fit(rush_model, train_loader, val_loader)

    # remove the temp dir
    shutil.rmtree(TEMP_LOG_DIR)
