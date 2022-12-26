from abc import abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchrush.module.base import BaseModule


class LeNet(BaseModule):
    def _init_model(self, input_size: Optional[Tuple] = None, embedding_size: Optional[int] = 84):
        if input_size is None:
            input_size = (1, 28, 28)
            in_channels = 1
        else:
            in_channels = input_size[-1]
        self.input_size = input_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(nn.Linear(256, 120), nn.ReLU())
        self.fc2 = nn.Linear(120, embedding_size)

    def forward(self, x):
        """
        One forward pass through the network.

        Args:
            x: input
        """
        if len(self.input_size) == 1:
            return x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    @abstractmethod
    def compute_loss(self, y_pred, y_true):
        pass

    def shared_step(self, batch, batch_idx, mode="train"):
        # preprocess
        x, y = batch

        # forward
        logits = self(x)

        # compute loss
        loss = self.compute_loss(logits, y)

        # prepare output
        return {"loss": loss, "predictions": torch.argmax(logits, -1), "references": y}


class LeNetForClassification(LeNet):
    def _init_model(
        self,
        input_size: Optional[Tuple] = None,
        embedding_size: Optional[int] = 84,
        output_size: Optional[int] = 10,
    ):
        super(LeNetForClassification, self)._init_model(input_size, embedding_size)
        self.output_size = output_size
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(embedding_size, output_size))

    def forward(self, x):
        x = super(LeNetForClassification, self).forward(x)
        x = self.out(x)
        return x

    def compute_loss(self, y_pred, y_true):
        if y_true.ndim == 1:
            y_true = F.one_hot(y_true, self.output_size) * 1.0
        return self.criterion(y_pred, y_true)
