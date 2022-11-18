from typing import Any, Dict
from pytorch_lightning.loggers import (
    NeptuneLogger,
    WandbLogger,
    TensorBoardLogger,
    MLFlowLogger,
    CSVLogger,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class NeptuneLogger(NeptuneLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any]):
        for key, value in any.items():
            self.experiment[key].log(value)


class WandbLogger(WandbLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any]):
        self.experiment.log(any)


class TensorBoardLogger(TensorBoardLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any]):
        for key, value in any.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value)
            elif isinstance(value, str):
                self.writer.add_text(key, value)
            elif isinstance(value, dict):
                self.writer.add_scalars(key, value)


class MLFlowLogger(MLFlowLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any]):
        self.log_metrics(any)


class CSVLogger(CSVLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any]):
        self.experiment.log_metrics(any)
        self.save()
