from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, embedding_size: int, nclass: int, apply_softmax: bool = False):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.classification_head = nn.Linear(embedding_size, nclass)

    def forward(self, x):
        x = self.classification_head(x)
        if self.apply_softmax:
            x = nn.Softmax()(x)
        return x
