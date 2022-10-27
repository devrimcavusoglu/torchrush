from abc import abstractmethod
from typing import Optional, Tuple

from torch import nn
import torch.nn.functional as F

from torchrush.model.base import BaseModule


class LeNet(BaseModule):
    def _init(self, input_size: Optional[Tuple] = None, embedding_size: Optional[int] = 84):
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

    def _forward(self, x):
        """
        One forward pass through the network.

        Args:
            x: input
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    @abstractmethod
    def compute_loss(self, y_pred, y_true):
        pass

    def preprocess(self, x):
        if len(self.input_size) == 1:
            return x.view(x.size(0), -1)
        return x


class LeNetForClassification(LeNet):
    def _init(self, input_size: Optional[Tuple] = None, embedding_size: Optional[int] = 84, output_size: Optional[int] = 10):
        super(LeNetForClassification, self)._init(input_size, embedding_size)
        self.output_size = output_size
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(embedding_size, output_size))

    def _forward(self, x):
        x = super(LeNetForClassification, self)._forward(x)
        x = self.out(x)
        return x

    def compute_loss(self, y_pred, y_true):
        if y_true.ndim == 1:
            y_true = F.one_hot(y_true, self.output_size) * 1.0
        return self.criterion(y_pred, y_true)

