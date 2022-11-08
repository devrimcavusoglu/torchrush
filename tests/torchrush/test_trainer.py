import pytest
import pytorch_lightning as pl

from torchrush.data_loader import DataLoader
from torchrush.dataset import GenericImageClassificationDataset
from torchrush.model.lenet5 import LeNetForClassification


@pytest.fixture(scope="function")
def pl_trainer():
    return pl.Trainer(enable_checkpointing=False, max_epochs=1)


@pytest.fixture(scope="function")
def rush_model():
    return LeNetForClassification(criterion="CrossEntropyLoss", optimizer="SGD", input_size=(28, 28, 1), lr=0.01)


@pytest.fixture(scope="function")
def data_loaders():
    train_loader = DataLoader.from_datasets(
        "mnist", split="train", constructor=GenericImageClassificationDataset, batch_size=32
    )
    val_loader = DataLoader.from_datasets(
        "mnist", split="test", constructor=GenericImageClassificationDataset, batch_size=32
    )
    return train_loader, val_loader


def test_trainer_trains(pl_trainer, rush_model, data_loaders):
    train_loader, val_loader = data_loaders
    pl_trainer.fit(rush_model, train_loader, val_loader)
