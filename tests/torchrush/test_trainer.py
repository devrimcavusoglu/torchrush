import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from torchrush.data_loader import DataLoader
from torchrush.dataset import GenericImageClassificationDataset
from torchrush.module.base import BaseModule
from torchrush.module.lenet5 import LeNetForClassification


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


def test_trainer_trains(rush_model, data_loaders, loss_callback):
    train_loader, val_loader = data_loaders
    pl_trainer = pl.Trainer(enable_checkpointing=False, max_steps=200, callbacks=[loss_callback])
    pl_trainer.fit(rush_model, train_loader, val_loader)
    losses = loss_callback.losses
    assert list(sorted(losses, reverse=True)) == losses
