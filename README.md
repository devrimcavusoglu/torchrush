<p align="center">
  <img src="./rsc/logo.png" width="200" height="200" />
</p>

<h1 align="center">TorchRush</h1>
Yet another framework built on top of PyTorch that is designed with a high speed experimental setup in the mind.

## Setup

Create a conda environment and install required packages

    conda env create -f environment.yml

Set up development with the following command (under project root directory), (you can use your IDE settings as well if supporting conda environments)

    conda develop .

## Training

Training is quite straightforward with CLI or configuration file (json). You can see the help doc with the following commands

    python stact/train.py from_cli --help

or

    python stact/train.py from_config --help

Example training config is like follows, it includes all required construction and training arguments.

```json
{
  "model_name": "ClassicMNISTModel",
  "model_collection": "mnist",
  "dataset": "mnist",
  "criterion": "CrossEntropyLoss",
  "lr": 0.01,
  "epochs": 10,
  "output_path": "default",
  "exclude_validation": false,
  "device": "cuda",
  "optimizer": "SGD",
  "weight_decay": 0,
  "train_batch_size": 32,
  "validation_batch_size": 32,
  "num_workers": 1
}
```

To log the metrics into a neptune run, you need to also specify `neptune_config` argument which is path to the neptune configuration file (json). If not given, the trainer assumes the neptune config lies under `<PROJECT_ROOT>/cfg/neptune.json`, and tries to read from there. Example neptune config file.

```json
{
    "project": "<neptune_project_name>",
    "api_token": "<api_token>"
}
```

After the training is completed, the model is saved to `outputs/` directory (it will be created if it doesn't exist). A unique directory will be created for the model, and the files (weights, jsons, etc.) are saved into this folder.

## Prediction

WIP.
