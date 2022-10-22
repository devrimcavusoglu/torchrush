from typing import Dict

from torch import nn

from stact import StAct
from stact.models.base import BaseConvolutionalNetwork


class LeNet(BaseConvolutionalNetwork):
    def _init(self, output_size: int = 10):
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(6),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(256, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, output_size)

    def forward(self, x):
        '''
        One forward pass through the network.

        Args:
            x: input
        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class StActLeNet(LeNet):
    def _init(self, activation: Dict[str, float] = None, output_size: int = 10):
        super()._init(output_size=output_size)

        if activation is None:
            activation = {"ReLU": 0.5, "Tanh": 0.5}
        self.activation = activation

        self.stact1 = StAct(256, 120, self.activation, mode="s2m")
        self.stact2 = StAct(self.stact1, 84, self.activation, mode="m2m")
        self.stact_output = StAct(self.stact2, output_size, self.activation, mode="m2s")

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.flatten(start_dim=1)
        x = self.stact1(x)
        x = self.stact2(x)
        x = self.stact_output(x)
        return x
