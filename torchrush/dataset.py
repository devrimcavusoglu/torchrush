from typing import Dict, List, Tuple, Union

import datasets
import torch
from torch.utils.data import Dataset

from torchrush.exceptions import DatasetNotLoadedError
from torchrush.utils.data import pillow_to_torch


class GenericDataset(Dataset):
    def __init__(self, name: str, split: str, load_on_init: bool = True):
        self.name = name
        self.split = split
        self._dataset = None
        if load_on_init:
            self.load()

    def __getitem__(self, subscript: Union[int, List[int], slice]):
        if self._dataset is None:
            raise DatasetNotLoadedError()

        if isinstance(subscript, slice):
            start = subscript.start or 0
            stop = subscript.stop or len(self)
            step = subscript.step or 1
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(subscript, list):
            return [self[i] for i in subscript]
        elif not isinstance(subscript, int):
            raise TypeError(f"Expected type of int, got {type(subscript)}.")
        return self.get_item(subscript)

    def __len__(self):
        if self._dataset is None:
            raise DatasetNotLoadedError()
        return len(self._dataset)

    @property
    def input_size(self):
        return self[0][0].shape

    def get_item(self, index: int) -> Dict:
        return self._dataset[index]

    def load(self):
        if self._dataset is None:
            self._dataset = datasets.load_dataset(self.name, split=self.split)


class GenericImageClassificationDataset(GenericDataset):
    def get_item(self, index) -> Tuple[torch.Tensor, int]:
        col_image, col_label = self._dataset.column_names
        return (
            pillow_to_torch(self._dataset[index][col_image]),
            self._dataset[index][col_label],
        )
