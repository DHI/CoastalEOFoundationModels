"""TerraTorch's `SemanticSegmentationTask` with test-time augmentation and extra test metrics."""

from pathlib import Path
from typing import Any

import torch
from terratorch.models.model import ModelOutput
from terratorch.tasks import SemanticSegmentationTask
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
)

# The MADOS reference's eight TTA views (`utils/test_time_aug.py`): four rotations, each
# with and without a horizontal flip.
TTA_VIEWS = tuple((k, flip) for k in range(4) for flip in (False, True))

TTA_AGGREGATIONS = ("vote", "mean_logits")


class AquaSegmentationTask(SemanticSegmentationTask):
    """Segmentation task with optional test-time augmentation and extra test outputs.

    Args:
        tta: predict on the eight reference views at test time and combine them.
        tta_aggregation: `"vote"` takes the per-pixel majority over the views' label maps,
            as the reference does; `"mean_logits"` averages their logits.
        extended_per_class_metrics: also log per-class precision and F1.
        confusion_matrix: write the test confusion matrix to `confusion_matrix_test.csv`.
    """

    def __init__(
        self,
        *args: Any,
        tta: bool = False,
        tta_aggregation: str = "vote",
        extended_per_class_metrics: bool = False,
        confusion_matrix: bool = False,
        **kwargs: Any,
    ) -> None:
        if tta_aggregation not in TTA_AGGREGATIONS:
            raise ValueError(f"tta_aggregation={tta_aggregation!r}; use one of {TTA_AGGREGATIONS}")
        # The parent's save_hyperparameters also stores the arguments above.
        super().__init__(*args, **kwargs)
        if tta and self.hparams.get("tiled_inference_on_testing"):
            raise ValueError("tta does not support tiled_inference_on_testing; disable one of them")
        self._confusion_matrix = None

    def configure_metrics(self) -> None:
        super().configure_metrics()
        if not self.hparams["extended_per_class_metrics"]:
            return

        num_classes = self.hparams["model_args"]["num_classes"]
        ignore_index = self.hparams["ignore_index"]
        class_names = self.hparams["class_names"]
        # Per-class recall is already logged by the parent as `Class_Accuracy`.
        extra = MetricCollection(
            {
                "Precision": ClasswiseWrapper(
                    MulticlassPrecision(
                        num_classes=num_classes, ignore_index=ignore_index, average=None
                    ),
                    labels=class_names,
                    prefix="Precision_",
                ),
                "F1": ClasswiseWrapper(
                    MulticlassF1Score(
                        num_classes=num_classes, ignore_index=ignore_index, average=None
                    ),
                    labels=class_names,
                    prefix="F1_",
                ),
            }
        )
        self.train_metrics.add_metrics(extra.clone())
        self.val_metrics.add_metrics(extra.clone())
        for collection in self.test_metrics:
            collection.add_metrics(extra.clone())

    def _tta_forward(self, x: torch.Tensor, **rest: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(mean_logits, voted_labels)` over the eight views, mapped back to the input frame."""
        logit_sum = None
        per_view_labels = []

        for k, flip in TTA_VIEWS:
            view = torch.rot90(x, k=k, dims=(-2, -1))
            if flip:
                view = torch.flip(view, dims=(-1,))

            logits = self(view, **rest).output

            if flip:
                logits = torch.flip(logits, dims=(-1,))
            logits = torch.rot90(logits, k=-k, dims=(-2, -1))

            logit_sum = logits if logit_sum is None else logit_sum + logits
            per_view_labels.append(logits.argmax(dim=1))

        mean_logits = logit_sum / len(TTA_VIEWS)
        voted = torch.mode(torch.stack(per_view_labels), dim=0).values
        return mean_logits, voted

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not (self.hparams["tta"] or self.hparams["confusion_matrix"]):
            return super().test_step(batch, batch_idx, dataloader_idx)

        x = batch["image"]
        y = self.squeeze_ground_truth(batch["mask"])
        rest = {k: batch[k] for k in batch.keys() - {"image", "mask", "filename"}}

        if self.hparams["tta"]:
            mean_logits, voted = self._tta_forward(x, **rest)
            model_output = ModelOutput(output=mean_logits)
            y_hat_hard = voted if self.hparams["tta_aggregation"] == "vote" else mean_logits.argmax(1)
        else:
            model_output = self.handle_full_or_tiled_inference(
                x, self.tiled_inference_on_testing, **rest
            )
            y_hat_hard = model_output.output.argmax(dim=1)

        if dataloader_idx >= len(self.test_loss_handler):
            raise ValueError("more test dataloaders than test_dataloaders_names; add a name per dataloader")
        # With TTA the loss uses the mean logits, whichever aggregation picks the labels.
        loss = self.test_loss_handler[dataloader_idx].compute_loss(
            model_output, y, self.criterion, self.aux_loss
        )
        self.test_loss_handler[dataloader_idx].log_loss(
            self.log, loss_dict=loss, batch_size=y.shape[0]
        )
        self.test_metrics[dataloader_idx].update(y_hat_hard, y)
        self.record_metrics(dataloader_idx, y_hat_hard, y)

        if self.hparams["confusion_matrix"]:
            self._update_confusion_matrix(y_hat_hard, y)

        if self._do_plot_samples(batch_idx):
            batch["prediction"] = y_hat_hard
            self.plot_sample(batch, batch_idx)

    def _update_confusion_matrix(self, y_hat_hard: torch.Tensor, y: torch.Tensor) -> None:
        if self._confusion_matrix is None:
            # Not in the MetricCollection, whose values are logged as scalars.
            self._confusion_matrix = MulticlassConfusionMatrix(
                num_classes=self.hparams["model_args"]["num_classes"],
                ignore_index=self.hparams["ignore_index"],
            ).to(y.device)
        self._confusion_matrix.update(y_hat_hard, y)

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        if self._confusion_matrix is None:
            return

        matrix = self._confusion_matrix.compute().cpu().numpy()
        self._confusion_matrix.reset()
        names = self.hparams["class_names"] or [
            str(i) for i in range(self.hparams["model_args"]["num_classes"])
        ]
        # Rows are ground truth, columns are predictions (torchmetrics convention).
        lines = ["true\\predicted," + ",".join(names)]
        lines += [f"{name}," + ",".join(str(int(v)) for v in row) for name, row in zip(names, matrix)]

        directory = Path(self.trainer.log_dir or ".") if self.trainer is not None else Path(".")
        path = directory / "confusion_matrix_test.csv"
        path.write_text("\n".join(lines) + "\n")
        print(f"Wrote confusion matrix: {path}")
