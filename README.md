<p align="center">
  <img src="./rsc/logo.png" width="200" height="200" />
</p>

<h1 align="center">TorchRush</h1>

Powerful torch based deep-learning framework with combination of tools from PyTorch-Lightning and HuggingFace.

Yet another framework built on top of PyTorch that is designed with a high speed experimental setup in mind. It also 
possesses the power of allowing you to build and alter the core of the building blocks of your development and/or 
research projects. 

# Installation

Install with

```shell
git clone git@github.com:devrimcavusoglu/torchrush.git
cd torchrush
pip install -e .[dev]
```

# Training

Training is easy as follows.

```python
import pytorch_lightning as pl

from torchrush.data_loader import DataLoader
from torchrush.dataset import GenericImageClassificationDataset
from torchrush.module.lenet5 import LeNetForClassification

# Prepare datasets
train_loader = DataLoader.from_datasets(
		"mnist", split="train", constructor=GenericImageClassificationDataset, batch_size=32
)
val_loader = DataLoader.from_datasets(
		"mnist", split="test", constructor=GenericImageClassificationDataset, batch_size=32
)

# Set module
model = LeNetForClassification(criterion="CrossEntropyLoss", optimizer="SGD", input_size=(28, 28, 1), lr=0.01)

# Train
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, train_loader, val_loader)
```

# Experiment tracking

Logger classes should be imported from `torchrush.loggers` and metrics should be set using `torchrush.MetricCallback`:

```python
import pytorch_lightning as pl

from torchrush.loggers import TensorBoardLogger, NeptuneLogger
from torchrush.metrics import MetricCallback

metric_callback = MetricCallback(metrics=['accuracy', 'f1', 'precision', 'recall'])

trainer = pl.Trainer(
    max_epochs=10,
    check_val_every_n_epoch=1,
    val_check_interval=1.0,
    logger=[TensorBoardLogger(), NeptuneLogger()],
    callbacks=[metric_callback]
)
```

`metrics` variable in `MetricCallback` can include any [evaluate default metrics](https://huggingface.co/evaluate-metric) or custom metrics from [hf/spaces](https://huggingface.co/spaces).

# Contributing

This repo is developed and currently maintained by [@devrimcavusoglu](https://github.com/devrimcavusoglu) and [@fcakyon](https://github.com/fcakyon). We welcome any contribution, so do not hesitate :) 

Before opening a PR, run tests and reformat the code with:

```bash
python -m tests.run_tests -rx
python -m tests.run_code_style format
```
