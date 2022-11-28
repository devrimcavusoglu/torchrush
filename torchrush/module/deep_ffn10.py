from abc import abstractmethod

import numpy as np
import torch
from torch import nn
from torch.nn import ModuleList

from torchrush.module.base import BaseModule


class DeepFFN10(BaseModule):
    def _init_model(self, input_size: int, activation: str = "ReLU"):
        self.input_nodes = np.product(input_size).item()
        self.input = self.dense_block(self.input_nodes, 128, activation)
        self.dense_layers = ModuleList([self.dense_block(128, 128, activation) for _ in range(8)])

    @staticmethod
    def dense_block(in_features, out_features, f):
        if isinstance(f, str):
            f = getattr(nn, f)

        layer = torch.nn.Sequential(nn.Linear(in_features, out_features), f())

        return layer

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        x = self.input(x)
        for layer in self.dense_layers:
            x = layer(x)
        return x

    @abstractmethod
    def compute_loss(self, y_pred, y_true):
        pass


class DeepFFN10Classifier(BaseModule):
    def _init_model(self, input_size: int, activation: str = "ReLU", output_size: int = 10):
        self.feature_extractor = DeepFFN10(input_size, activation)
        self.classifier = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def compute_loss(self, y_pred, y_true):
        bs, dim = y_pred.shape
        if y_true.ndim == 1:
            y_true = torch.nn.functional.one_hot(y_true, dim) * 1.0
        return self.criterion(y_pred, y_true)
