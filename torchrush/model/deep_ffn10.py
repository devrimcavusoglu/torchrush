from typing import Dict

import torch
from torch import nn

from torchrush.model.base import BaseFeedForwardNetwork


class DeepFFN10(BaseFeedForwardNetwork):
    def _init(self, activation: str = "ReLU", output_size: int = 10):
        self.input = self.dense_block(
            self.input_nodes, 128, activation
        )
        self.dense1 = self.dense_block(128, 128, activation)
        self.dense2 = self.dense_block(128, 128, activation)
        self.dense3 = self.dense_block(128, 128, activation)
        self.dense4 = self.dense_block(128, 128, activation)
        self.dense5 = self.dense_block(128, 128, activation)
        self.dense6 = self.dense_block(128, 128, activation)
        self.dense7 = self.dense_block(128, 128, activation)
        self.dense8 = self.dense_block(128, 128, activation)
        self.output = torch.nn.Linear(128, output_size)

    @staticmethod
    def dense_block(in_features, out_features, f):
        if isinstance(f, str):
            f = getattr(nn, f)

        layer = torch.nn.Sequential(nn.Linear(in_features, out_features), f())

        return layer

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.input(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        x = self.dense8(x)

        x = self.output(x)
        return x
