from typing import Optional

from torch.utils.data import DataLoader as _DataLoader

from torchrush.dataset import GenericDataset


class DataLoader(_DataLoader):
    @classmethod
    def from_datasets(
        cls,
        name_or_path: str,
        split: str,
        load_on_init: bool = True,
        constructor: Optional[type] = None,
        **kwargs,
    ):
        if constructor is None:
            constructor = GenericDataset

        if cls._validate_constructor(constructor):
            raise TypeError(
                f"Construct must be a subclass of "
                f"`torchrush.dataset.GenericDataset`, got `{type(constructor)}`."
            )
        dataset = constructor(name_or_path, split, load_on_init)
        return cls(dataset, **kwargs)

    @staticmethod
    def _validate_constructor(constructor: type) -> bool:
        if isinstance(constructor, GenericDataset):
            return True
        return False
