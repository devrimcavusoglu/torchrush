import pytest
import pytorch_lightning as pl

from torchrush.data_loader import DataLoader
from torchrush.dataset import GenericImageClassificationDataset
from torchrush.module.deep_ffn10 import DeepFFN10Classifier


@pytest.fixture
def trainer():
    return pl.Trainer(enable_checkpointing=False, max_steps=10)


@pytest.fixture(scope="function")
def data_loaders():
    train_loader = DataLoader.from_datasets(
        "mnist", split="train", constructor=GenericImageClassificationDataset, batch_size=32
    )
    val_loader = DataLoader.from_datasets(
        "mnist", split="test", constructor=GenericImageClassificationDataset, batch_size=32
    )
    return train_loader, val_loader


def test_module_construction_for_prediction(data_loaders):
    _, test_loader = data_loaders
    model = DeepFFN10Classifier(input_size=(1, 28, 28))
    x, y = test_loader.dataset[0]
    model(x)


@pytest.mark.parametrize(
    ("optimizer", "criterion"),
    [(None, None), ("SGD", None), (None, "CrossEntropyLoss"), ("SGD", "CrossEntropyLoss")],
)
def test_module_construction_for_training(optimizer, criterion, data_loaders, trainer):
    train_loader, test_loader = data_loaders
    if optimizer:
        # lr is a required param when optimizer is given.
        model = DeepFFN10Classifier(
            input_size=(1, 28, 28), optimizer=optimizer, criterion=criterion, lr=0.01
        )
    else:
        model = DeepFFN10Classifier(input_size=(1, 28, 28), optimizer=optimizer, criterion=criterion)
    if criterion is None:  # Because of validation sanity check of pl.Trainer
        with pytest.raises(ValueError):
            trainer.fit(model, train_loader, test_loader)
    elif optimizer is None:
        with pytest.raises(ValueError):
            trainer.fit(model, train_loader, test_loader)
    else:
        trainer.fit(model, train_loader, test_loader)
