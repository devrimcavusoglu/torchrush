from abc import abstractmethod

from torch import nn

from torchrush.model.base import BaseConvNet


class LeNet(BaseConvNet):
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
        self.fc1 = nn.Sequential(nn.Linear(256, 120), nn.ReLU())
        self.fc2 = nn.Linear(120, 84)

    def _forward(self, x):
        '''
        One forward pass through the network.

        Args:
            x: input
        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    @abstractmethod
    def compute_loss(self, y_pred, y_true):
        pass


class LeNetForClassification(LeNet):
    def _init(self, output_size: int = 10):
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(84, output_size))

    def _forward(self, x):
        x = super(LeNetForClassification, self)._forward(x)
        x = self.out(x)
        return x

    def compute_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)
