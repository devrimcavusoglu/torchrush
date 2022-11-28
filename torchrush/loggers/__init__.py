from typing import Any, Dict, Optional

from pytorch_lightning.loggers import CSVLogger as pl_CSVLogger
from pytorch_lightning.loggers import MLFlowLogger as pl_MLFlowLogger
from pytorch_lightning.loggers import NeptuneLogger as pl_NeptuneLogger
from pytorch_lightning.loggers import TensorBoardLogger as pl_TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger as pl_WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class NeptuneLogger(pl_NeptuneLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any], step: Optional[int] = None):
        for key, value in any.items():
            self.experiment[key].log(value, step=step)


class WandbLogger(pl_WandbLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any], step: Optional[int] = None):
        self.experiment.log(any, step=step)


class TensorBoardLogger(pl_TensorBoardLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any], step: Optional[int] = None):
        for key, value in any.items():
            if isinstance(value, (int, float)):
                self.experiment.add_scalar(key, value, global_step=step)
            elif isinstance(value, str):
                self.experiment.add_text(key, value, global_step=step)
            elif isinstance(value, dict):
                self.experiment.add_scalars(key, value, global_step=step)
        self.experiment.flush()


class MLFlowLogger(pl_MLFlowLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any], step: Optional[int] = None):
        self.log_metrics(any, step=step)


class CSVLogger(pl_CSVLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any], step: Optional[int] = None):
        self.experiment.log_metrics(any, step=step)
        self.save()
