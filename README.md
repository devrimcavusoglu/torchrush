<p align="center">
  <img src="./rsc/logo.png" width="200" height="200" />
</p>

<h1 align="center">TorchRush</h1>
Yet another framework built on top of PyTorch that is designed with a high speed experimental setup in the mind.

# Installation

Install with

```shell
git clone git@github.com:devrimcavusoglu/torchrush-dev.git
cd torchrush-dev
pip install -e .[dev]
```

# Training

Training is easy as follows.

```python
import pytorch_lightning as pl

from torchrush.data_loader import DataLoader
from torchrush.dataset import GenericImageClassificationDataset
from torchrush.model.lenet5 import LeNetForClassification


# Prepare datasets
train_loader = DataLoader.from_datasets(
        "mnist", split="train", constructor=GenericImageClassificationDataset, batch_size=32
)
val_loader = DataLoader.from_datasets(
    "mnist", split="test", constructor=GenericImageClassificationDataset, batch_size=32
)

# Set model
model = LeNetForClassification(criterion="CrossEntropyLoss", optimizer="SGD", input_size=(28, 28, 1), lr=0.01)

# Train
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, train_loader, val_loader)
```

# Contributing

Before opening a PR, run tests and reformat the code with:

```bash
python -m tests.run_tests
python -m tests.run_code_style format
```

# Experiment tracking

Logger classes should be imported from `torchrush.loggers`:

```python
from torchrush.loggers import TensorboardLogger, NeptuneLogger

trainer = pl.Trainer(max_epochs=1, logger=[TensorboardLogger(), NeptuneLogger()])
```