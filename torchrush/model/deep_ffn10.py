from abc import abstractmethod

import torch
from torch import nn
from torch.nn import ModuleList

from torchrush.model.base import BaseModule


class DeepFFN10(BaseModule):
    def _init_model(self, input_size: int, activation: str = "ReLU"):
        self.input = self.dense_block(self.input_nodes, 128, activation)
        self.dense_layers = ModuleList([self.dense_block(128, 128, activation) for _ in range(8)])

    @staticmethod
    def dense_block(in_features, out_features, f):
        if isinstance(f, str):
            f = getattr(nn, f)

        layer = torch.nn.Sequential(nn.Linear(in_features, out_features), f())

        return layer

    def _forward(self, x):
        x = self.input(x)
        for layer in self.dense_layers:
            x = layer(x)
        return x

    @abstractmethod
    def compute_loss(self, y_pred, y_true):
        pass


class DeepFFN10Classifier(DeepFFN10):
    def _init_model(self, input_size: int, activation: str = "ReLU", output_size: int = 10):
        super()._init_model(input_size, activation)
        self.output_size = output_size
        self.out = nn.Linear(128, output_size)

    def _forward(self, x):
        x = super(DeepFFN10Classifier, self)._forward(x)
        x = self.out(x)
        return x

    def compute_loss(self, y_pred, y_true):
        if y_true.ndim == 1:
            y_true = torch.nn.functional.one_hot(y_true, self.output_size) * 1.0
        return {"loss": self.criterion(y_pred, y_true), "predictions": y_pred, "references": y_true}
