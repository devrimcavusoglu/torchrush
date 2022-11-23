from typing import Any, Dict, Optional

from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, NeptuneLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class NeptuneLogger(NeptuneLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any], step: Optional[int] = None):
        for key, value in any.items():
            self.experiment[key].log(value, step=step)


class WandbLogger(WandbLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any], step: Optional[int] = None):
        self.experiment.log(any, step=step)


class TensorBoardLogger(TensorBoardLogger):
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


class MLFlowLogger(MLFlowLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any], step: Optional[int] = None):
        self.log_metrics(any, step=step)


class CSVLogger(CSVLogger):
    @rank_zero_only
    def log_any(self, any: Dict[str, Any], step: Optional[int] = None):
        self.experiment.log_metrics(any, step=step)
        self.save()
