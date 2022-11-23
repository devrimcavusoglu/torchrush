from typing import Any, List

import evaluate
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.model_summary import summarize


class CombinedEvaluations:
    # place holder for evaluate.combine till https://github.com/huggingface/evaluate/issues/234 is fixed
    def __init__(self, metrics: List[str]):
        self.metrics = [evaluate.load(metric) if isinstance(metric, str) else metric for metric in metrics]

    def add_batch(self, predictions: Any, references: Any):
        for metric in self.metrics:
            metric.add_batch(predictions=predictions, references=references)

    def compute(self, **kwargs):
        results = {}
        zero_division = kwargs.get("zero_division", "warn")
        kwargs.pop("zero_division")
        for metric in self.metrics:
            if metric.name == "precision":
                results.update(metric.compute(zero_division=zero_division, **kwargs))
            else:
                results.update(metric.compute(**kwargs))
        return results


class MetricCallback(Callback):
    def __init__(
        self,
        eval_freq: int = 1,
        metric_list: List[str] = None,
        labels: List[str] = None,
        log_labelwise_metrics: bool = False,
        log_on: str = "epoch_end",
    ):
        """Init metrics on train calss init."""
        super().__init__()
        self.eval_freq = eval_freq
        self.combined_evaluations = {
            "train": CombinedEvaluations(metric_list),
            "val": CombinedEvaluations(metric_list),
            "test": CombinedEvaluations(metric_list),
        }

        if log_labelwise_metrics and labels is None:
            raise ValueError("If `log_labelwise_metrics` is True, `labels` must be provided.")

        self.labels = labels
        self.log_labelwise_metrics = log_labelwise_metrics
        self.log_on = log_on

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Log config and model summary on train start."""
        model_summary = summarize(pl_module)
        pl_module.log_any({"model_stats/size": model_summary.model_size})
        pl_module.log_any({"model_stats/total_parameters": model_summary.total_parameters})
        pl_module.log_any({"model_stats/trainable_parameters": model_summary.trainable_parameters})

        pl_module.log_hyperparams()

    def _accumulate_metrics(self, outputs, mode: str = "train"):
        if mode not in ["train", "val", "test"]:
            raise ValueError("`mode` must be one of {'train', 'val', 'test'}.")

        # accumulate supervised metrics if possible
        if "predictions" and "references" in outputs:
            self.combined_evaluations[mode].add_batch(
                predictions=outputs["predictions"], references=outputs["references"]
            )

    def _log_metrics(self, step: int, pl_module: "pl.LightningModule", mode: str = "train"):
        if mode not in ["train", "val", "test"]:
            raise ValueError("`mode` must be one of {'train', 'val', 'test'}.")

        # log supervised metrics if possible
        result = self.combined_evaluations.compute(average=None, zero_division=0)
        for metric, scores in result.items():
            # log macro score
            pl_module.log_any({f"val/{metric}": scores.mean()}, step=step)
            # log label-wise scores
            if self.log_labelwise_metrics:
                for label, score in zip(self.labels, scores):
                    pl_module.log_any(
                        {f"val/{metric}_{label}": score},
                        step=step,
                    )
        self._last_val_result = result

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int
    ) -> None:
        # accumulate metrics
        self._accumulate_metrics(outputs, mode="train")

        # log metrics
        pl_module.log_any({"train/loss": outputs["loss"].item()}, step=batch_idx)
        if batch_idx % self.eval_freq == 0 and self.log_on == "batch_end":
            self._log_metrics(batch_idx, pl_module, mode="train")

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        # accumulate metrics
        self._accumulate_metrics(outputs, mode="val")

        # log metrics
        pl_module.log_any({"val/loss": outputs["loss"].item()}, step=batch_idx)
        if batch_idx % self.eval_freq == 0 and self.log_on == "batch_end":
            self._log_metrics(batch_idx, pl_module, mode="val")

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        # accumulate metrics
        self._accumulate_metrics(outputs, mode="test")

        # log metrics
        pl_module.log_any({"test/loss": outputs["loss"].item()}, step=batch_idx)
        if batch_idx % self.eval_freq == 0 and self.log_on == "batch_end":
            self._log_metrics(batch_idx, pl_module, mode="test")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.log_on == "epoch_end" and trainer.current_epoch % self.eval_freq == 0:
            self._log_metrics(trainer.current_epoch, pl_module, mode="train")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.log_on == "epoch_end" and trainer.current_epoch % self.eval_freq == 0:
            self._log_metrics(trainer.current_epoch, pl_module, mode="val")

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.log_on == "epoch_end" and trainer.current_epoch % self.eval_freq == 0:
            self._log_metrics(trainer.current_epoch, pl_module, mode="test")