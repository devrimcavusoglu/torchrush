from torch import nn
from torch.nn import ModuleList
import torch.nn.functional as F

from torchrush.model.base import BaseMLP
from torchrush.model.classification_utils import ClassificationHead


class MyModel(BaseMLP):
    def _init(self, *args, **kwargs):
        self.input_layer = nn.Sequential(nn.Linear(self.input_shape, 128), nn.ReLU())
        self.hidden_layers = ModuleList([nn.Sequential(
                nn.Linear(128, 128), nn.ReLU()
        ) for _ in range(10)])
        self.output_layer = ClassificationHead(128, 10)

    def _forward(self, x):
        x = self.input_layer(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x

    def compute_loss(self, y_pred, y_true):
        if y_true.ndim == 1:
            y_true = F.one_hot(y_true, self.output_size) * 1.0
        return self.criterion(y_pred, y_true)


if __name__ == "__main__":
    import pytorch_lightning as pl
    from torchrush.data_loader import DataLoader
    from torchrush.dataset import GenericImageClassificationDataset

    model = MyModel(input_size=(28, 28), output_size=10, criterion="CrossEntropyLoss", optimizer="SGD", lr=0.01)

    train_loader = DataLoader.from_datasets("mnist", split="train", constructor=GenericImageClassificationDataset, batch_size=16)
    validation_loader = DataLoader.from_datasets("mnist", split="test", constructor=GenericImageClassificationDataset, batch_size=16)

    trainer = pl.Trainer()
    trainer.fit(model, train_loader, validation_loader)
