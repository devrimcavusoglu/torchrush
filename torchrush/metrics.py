from typing import Any, Dict, List

import evaluate
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.model_summary import summarize

LABELWISE_SUPPORTED_METRICS = [
    "precision",
    "recall",
    "f1",
]


class CombinedEvaluations:
    # place holder for evaluate.combine till https://github.com/huggingface/evaluate/issues/234 is fixed
    def __init__(self, metrics: List[str]):
        self.metrics = [
            evaluate.load(metric) if isinstance(metric, str) else metric for metric in metrics
        ]

    def add_batch(self, predictions: Any, references: Any) -> None:
        for metric in self.metrics:
            metric.add_batch(predictions=predictions, references=references)

    def compute(self, labelwise: bool = False, **kwargs) -> Dict[str, float]:
        results = {}

        zero_division = kwargs.get("zero_division", 0)
        if kwargs.get("zero_division") is not None:
            kwargs.pop("zero_division")

        average = kwargs.get("average", "macro") if not labelwise else None
        if kwargs.get("average") is not None:
            kwargs.pop("average")

        for metric in self.metrics:
            if metric.name == "precision":
                results.update(metric.compute(zero_division=zero_division, average=average, **kwargs))
            elif metric.name in LABELWISE_SUPPORTED_METRICS:
                results.update(metric.compute(average=average, **kwargs))
            else:
                results.update(metric.compute(**kwargs))
        return results


class MetricCallback(Callback):
    def __init__(
        self,
        metrics: List[str] = None,
        labels: List[str] = None,
        log_labelwise_metrics: bool = False,
    ):
        super().__init__()
        self.combined_evaluations = {
            "train": CombinedEvaluations(metrics),
            "val": CombinedEvaluations(metrics),
            "test": CombinedEvaluations(metrics),
        }

        if log_labelwise_metrics and labels is None:
            raise ValueError("If `log_labelwise_metrics` is True, `labels` must be provided.")

        self.labels = labels
        self.log_labelwise_metrics = log_labelwise_metrics
        self._test_loss = []
        self._val_loss = []
        self._last_train_step = 0

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Log config and module summary on train start."""
        model_summary = summarize(pl_module)
        pl_module.log_any({"model_stats/size": model_summary.model_size})
        pl_module.log_any({"model_stats/total_parameters": model_summary.total_parameters})
        pl_module.log_any({"model_stats/trainable_parameters": model_summary.trainable_parameters})

        pl_module.log_hyperparams()

    def _add_batch(self, outputs: Dict[str, Any], mode: str = "train") -> None:
        if mode not in ["train", "val", "test"]:
            raise ValueError("`mode` must be one of {'train', 'val', 'test'}.")

        # accumulate metrics if possible
        if "predictions" and "references" in outputs:
            self.combined_evaluations[mode].add_batch(
                predictions=outputs["predictions"], references=outputs["references"]
            )
        loss = outputs["loss"]
        if mode == "val":
            self._val_loss.append(loss)
        elif mode == "test":
            self._test_loss.append(loss)

    def _compute_loss(self, mode: str = "val") -> float:
        if mode == "val":
            avg_loss = sum(self._val_loss) / len(self._val_loss)
            self._val_loss = []
        elif mode == "test":
            avg_loss = sum(self._test_loss) / len(self._test_loss)
            self._test_loss = []
        else:
            raise ValueError("Train losses are not aggregated.")
        return avg_loss

    def _log_metrics(self, step: int, pl_module: "pl.LightningModule", mode: str = "train") -> None:
        if mode not in ["train", "val", "test"]:
            raise ValueError("`mode` must be one of {'train', 'val', 'test'}.")

        # log metrics if possible
        result = self.combined_evaluations[mode].compute(labelwise=self.log_labelwise_metrics)
        for metric, scores in result.items():
            try:  # calculate macro score if labelwise scores are available
                overall_score = scores.mean().item()
                is_labelwise_metric = True
            except AttributeError:
                overall_score = scores
                is_labelwise_metric = False

            # log overall score
            pl_module.log_any({f"{mode}/{metric}": overall_score}, step=step)

            # log label-wise scores
            if self.log_labelwise_metrics and is_labelwise_metric:
                for label, score in zip(self.labels, scores):
                    pl_module.log_any(
                        {f"{mode}/{metric}_{label}": score},
                        step=step,
                    )
        self._last_val_result = result

    def _should_log_metrics(self, trainer: "pl.Trainer") -> bool:
        return not trainer.sanity_checking and trainer.fit_loop.epoch_loop._should_check_val_fx()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._add_batch(outputs, mode="train")

        idx = trainer.current_epoch * trainer.num_training_batches + batch_idx

        # log metrics
        pl_module.log_any({"train/loss": outputs["loss"]}, step=idx)
        if self._should_log_metrics(trainer):
            self._log_metrics(idx, pl_module, mode="train")
        self._last_train_step = idx

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._add_batch(outputs, mode="val")

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._add_batch(outputs, mode="test")

    def on_evaluation_end(self, trainer, pl_module: "pl.LightningModule", mode: str = "val") -> None:
        avg_eval_loss = self._compute_loss(mode)
        # log metrics
        pl_module.log_any({f"{mode}/loss": avg_eval_loss}, step=self._last_train_step)
        if self._should_log_metrics(trainer):
            self._log_metrics(self._last_train_step, pl_module, mode=mode)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_evaluation_end(trainer, pl_module, mode="val")

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_evaluation_end(trainer, pl_module, mode="test")
